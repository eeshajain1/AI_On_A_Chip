import math


#CNN Layer Configs, (num_filters (output channels), num_channels (input channels), kernel_h, kernel_w), only consider the convolutional layers for our calcualtions
layers = [
    (32, 3, 3, 3),
    (64, 32, 3, 3),
    (128, 64, 3, 3),
    (256, 128, 3, 3),
    (512, 256, 3, 3)
]

#output spatial dimensions after each layer (with pooling after layers 2 and 4)
#remember its 32x32 32x32 16x16 16x16 16x16 for the height and width 
image_dims = [32, 32, 16, 16, 16]

#system specs
activation_bitwidth = 8
weight_bitwidth = 8 
DRAM_access_energy = 4e-12 #4 pj/bit
SRAM_access_energy = 0.1e-12 #0.1 pj/bit
single_DPU_energy_per_cycle = 20e-12 #20pJ
clock_period = 1e-9 #1 ns

# All latency number here are number of cycles required to do the operation. And for memory
# load it means the latency to read the all required data from the SRAM to buffer, same for
# memory write. Computation latency means the latency required to perform one dot product
# operation for all 16 dot product engines in parallel. DRAM latency means the latency required to
# load weights from all layers into weight SRAM.

DRAM_access_latency = 7000
weight_SRAM_load_latency = 6
activation_SRAM_load_latency = 3
activation_SRAM_write_latency = 3
compute_latency = 2

n_dot_product_units = 16
dot_product_unit_size = 128

#1a
def compute_layer_energy(dataflow, batch_size):
    total_energy = 0

    for i, (n_filter, n_channel, kernel_size, S) in enumerate(layers):
        H = W = image_dims[i]
        

        if dataflow == 'IS':  #Input Stationary
    
            
  
            
   
            #the number of times the input buffers have to be updated is just the number of inputs over the dot product unit size
            #recall that each dot product unit will have the same input vector as this vector is broadcasted to all the DPUs and 
            #multiply by batch size bc we have to do this for every image
            NIBU = math.ceil(n_channel*H*W/dot_product_unit_size) * batch_size
            # NWBU = math.ceil(n_channel*kernel_size*kernel_size/dot_product_unit_size) * math.ceil(n_filter/n_dot_product_units) * batch_size
            # print(NWBU)
            
            #every time we update hte buffer for a valid output pixel, we want to do a dot product, and we do this for each image in the batch
            #each input can only effect 9 -- filter size
            #9 output pixels that depend on input pixel
            #each patch in the image has input pixel (assume adequate padding) and that input pixel is flattened for IM2COL kernel_size times, 
            #so every time the inputs update, the input pixels have been used for dot products kernel_size times 
            NDPC = NIBU * kernel_size * kernel_size
            
            ISE = (H*W*n_channel) * activation_bitwidth * SRAM_access_energy * batch_size
            print("ISE ", ISE)
            #ISE = dot_product_unit_size * NIBU * activation_bitwidth * SRAM_access_energy
            
            #weight buffer needs to be updated for every dot product unit computation (NDPC)
            WSE = NDPC * dot_product_unit_size * weight_bitwidth * SRAM_access_energy 
            print("DES ", WSE)
            
            #dot product size is the size of the flattened filter: input_channels x kernel_size x kernel_size
            #and only the first dot product does not participate in the accumulation
            flat_dot_product_size = n_channel * kernel_size * kernel_size
            OSWE = NDPC * n_dot_product_units * activation_bitwidth * SRAM_access_energy
            OSRE = ((flat_dot_product_size - dot_product_unit_size)/(flat_dot_product_size)) * OSWE
            OSE = OSRE + OSWE
            
            DPE = NDPC * n_dot_product_units * single_DPU_energy_per_cycle
            
            total_energy += WSE + ISE + OSE + DPE

            
            #of weights * dram access energy at the end ?

        elif dataflow == 'OS':  #Output Stationary
            #we do not need NOBU because partial sums are kept
            #local to the array and only written back to SRAM 
            # after accumulation is complete
            #NDPC is equal to the number of buffer updates (I think), because we are not really reloading the outputs except for when we tae a dot product
            NOBU = H * W #Num of output buffer updates 
            
            #math.ceil(n_channel * kernel_size * kernel_size / dot_product_unit_size) is how many times the dot product needs to 
            #be computed to produce one output, then you multiply by output size (H*W) and then by batch_size
            NDPC =  math.ceil(n_channel * kernel_size * kernel_size / dot_product_unit_size) * H * W * batch_size
            
            #NDPC = math.ceil(n_filter/n_dot_product_units) * H * W * batch_size #for each filter a full output is produced
            #both should both be the same because neither are stationary and are true for every output
            ISE = NDPC * dot_product_unit_size * activation_bitwidth * SRAM_access_energy
            WSE = NDPC * dot_product_unit_size * weight_bitwidth * SRAM_access_energy
            
            OSE = NDPC * activation_bitwidth * SRAM_access_energy  #only write it once, no need to read
            DPE = NDPC * n_dot_product_units * single_DPU_energy_per_cycle
            total_energy += ISE + WSE + OSE + DPE
            
    #repeat above for all layers then add the dram ready energy to move weights from dram to sram 
    #this is just done after all the final weights are calculated
    total_weights = n_filter * (n_channel * kernel_size * kernel_size)
    DRAM_energy = total_weights * weight_bitwidth * DRAM_access_energy  
    total_energy += DRAM_energy

    print(f"dataflow = {dataflow}, batch size = {batch_size}, energy = {total_energy} J")
#1b latency
def compute_latency_WS( batch_size):
    # DRAM_access_latency = 7000
    # weight_SRAM_load_latency = 6
    # activation_SRAM_load_latency = 3
    # activation_SRAM_write_latency = 3
    # compute_latency = 2
    total_latency = 0
    for i, (n_filter, n_channel, kernel_size, S) in enumerate(layers):
        H = W = image_dims[i]
        
        # All latency number here are number of cycles required to do the operation. And for memory
        # load it means the latency to read the all required data from the SRAM to buffer, same for
        # memory write. Computation latency means the latency required to perform one dot product
        # operation for all 16 dot product engines in parallel. DRAM latency means the latency required to
        # load weights from all layers into weight SRAM
        
        #NWBU is the amount of weight buffer updates needed and weight buffer updates can be loaded simultaneously from the SRAM
        #each weight only needs to be loaded once from the SRAM
        #use this value to calculate how many weights and inputs need to be loaded at once and then calculate the latency for that
        NWBU = math.ceil(n_filter / n_dot_product_units) * math.ceil(n_channel * kernel_size / dot_product_unit_size) * batch_size
        
        #this is the number of dot product computations needed 
        #we will calculate the latency later for every dot product
        NDPC = NWBU * H * W
      
        
        #get how much latency there is in accessing each weight buffer 
        #memory load is the latency to read all required data from SRAM into the buffer 
        #so just multiply the number of buffer updates by the weight sram load latency
        WS_parallel_latency = NWBU * weight_SRAM_load_latency
   
        #and then each input
        IS_parallel_latency = NDPC * activation_SRAM_load_latency
   

        #get the max bc Weights and activations can be loaded in parallel since there is no dependency
        WS_IS_latency = max(WS_parallel_latency, IS_parallel_latency)
        
        flat_dot_product_size = n_channel * kernel_size * kernel_size
        #the output buffer needs to write to the activation SRAM every time the DPU produces results 
        OSW_latency = NDPC * n_dot_product_units * activation_SRAM_write_latency
        #need to read the activations for all times except the first psum chunk where no accumulation is needed
        #cannot multiply by OSWE since we need to take into account the read vs write latency
        OSR_latency = ((flat_dot_product_size - dot_product_unit_size)/(flat_dot_product_size)) * NDPC * n_dot_product_units * activation_SRAM_load_latency
         
        #for the number of dot products done per dpu * dot product units is how many computations we need to do
        #Computation latency means the latency required to perform one dot product
        # operation for all 16 dot product engines in parallel
        DP_latency = NDPC * compute_latency
        
        total_latency += WS_IS_latency + OSW_latency + OSR_latency + DP_latency
    
    #remember that weights need to be loaded from the DRAM first into SRAM 
    #DRAM latency means the latency required to
    # load weights from all layers into weight SRAM
    DRAM_latency =  DRAM_access_latency  
    total_latency += DRAM_latency
    #total latency = cycles/operation
    #clock period = seconds / cycle
    latency_seconds = total_latency * clock_period
    FPS = batch_size / latency_seconds
   
    
    print(f"batch size = {batch_size}, latency = {latency_seconds} s, average FPS = {FPS}")
  
    return total_latency, FPS

        
        
        
    

#1a
compute_layer_energy('IS', 1)
compute_layer_energy('IS', 256)
compute_latency_WS(1)
compute_latency_WS(256)

compute_layer_energy('OS', 1)
compute_layer_energy('OS', 256)


