// per outplane
kernel void forward_fc_direct(const int inPlane,
		       const int outPlane, 
		       const int inSize,
		       global const float* images,
		       global const float* weights,
		       global const float* bias,
		       global float* output) {
  
  int gId = get_global_id(0); // index of the output array
  int inVol = inPlane * inSize * inSize;
  int iFilter = gId % outPlane;
  int isample = gId / outPlane;
  
  for(int pix = 0; pix < inVol; pix++){
      output[gId] += images[pix + isample*inVol] * weights[ pix * outPlane + iFilter];
  }
  
}


// per sample and outplane
kernel void forward_fc_block(const int inPlane,
			  const int ny,
			  const int inSize,
			  global const float* a,
			  global const float* b,
			  global const float* c,
			  global float* output,
			  local float* buf) {
  //   ii
  // jj
  int n = inPlane * inSize * inSize;
  int j = get_global_id(1);
  int ii= get_local_id(0);
  int jj = get_local_id(1);
  int iisize = get_local_size(0);
  float sum[256]={0};
  for (int l =0; l < ny ; l++ )
    for (int k =ii; k < n ; k += iisize ) sum[l] += a[k+j*n] * b[k*ny+l];
  
  for (int l =0; l < ny ; l++ ) buf[ (ii + iisize * jj)*ny + l ] = sum[l];
  
  barrier(CLK_LOCAL_MEM_FENCE);
  
  for (int l =0; l < ny ; l++ )
    for (int k = iisize >> 1; k; k >>= 1) {
      if (ii < k) buf[ (ii + jj * iisize)*ny +l ] += buf[ (ii+k + jj*iisize)*ny +l];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  
  if (ii==0){
     for (int l =0; l < ny ; l++ )
       output[j*ny+l] = buf[(jj*iisize)*ny +l];
  }
}

