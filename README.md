# Introduction
In my final project, I explore Trusted Execution environments on GPUs. A trusted execution environment is a secure area of a processor. It guarantees code and data loaded inside to be protected with respect to confidentiality and integrity. As my final project, I will create such an environment for Nvidia GPUs running Cuda. 

I did this by making a layer of abstractions over the default Cuda environment and making those abstractions secure. I also added some level of hardening against timing and prime and probe side-channel attacks.

I attempted to provide protection against PCI bus snooping attacks and protecting against an attack where the driver intercepts data being sent to the GPU

<img width="707" alt="visual" src="https://user-images.githubusercontent.com/22736920/131578028-71a6776a-8742-4216-a649-c8f0231f28a8.png">

# Basic principle
 I perform symmetric key encryption using AES before transferring those contents to the GPU. Now when those contents are accessed on the GPU, an AES decryption operation is performed. Hence we ensure that a man-in-the-middle of these transfers would not know the data contents that were transferred.
 
 # Abstractions and interesting files
 My main contribution is providing this secure interface in the form of an Array class. It makes development on Cuda a bit easier by handling memory allocations and provides security. 

`example.cu` is another interesting file because it's my efficient implementation of the K-means algorithm for a GPU. It uses architectural features of the GPU (many cores, shared memory cache) to create a faster algorithm than a simple concurrent program.
The AES encryption and decryption files are from a research project and were created by researchers to be secure against side-channel attacks.
 
```
class SecureCudaArray {
  set(host array) allocates space for the array, encrypts and copies host array to the GPU
  
  get_data() decrypts on the GPU and gets a device pointer to data
  
  get() encrypts on GPU and copies over data to the CPU where it's decrypted
}
```

# Testing
I created a Kmeans implementation for one of my projects for this class, I used that as a baseline and replaced the arrays with my secure arrays to get a comparison.

![Runtime](https://user-images.githubusercontent.com/22736920/131578473-5712c9eb-b2e2-423d-a24c-b176a12dccc8.png)

As you can see, the overhead of the secure operations is not much—low enough to be used practically.
