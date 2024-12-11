#write a program that checks for multiple cuda devices and prints out their names
import torch

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    #print(torch.cuda.get_device_name(1))
else:

    print("No CUDA devices available")

# If a CUDA device is available, the program will print out the names of the available devices. If no CUDA devices are available, it will print "No CUDA devices available".
