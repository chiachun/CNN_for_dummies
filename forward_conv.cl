kernel void forward_conv_quick(const int nPlane,
		     const int nFilter, 
		     const int filterSize,
		     const int stride,
		     const int padding,
		     const int inSize,
		     const int outSize,
		     global const float* images,
		     global const float* filter,
		     global const float* bias,
		     global float* output) {
  
  int gId = get_global_id(0); // index of the output array
  int inSize2 = inSize * inSize;
  int outSize2 = outSize * outSize;
  int filterSize2 = filterSize * filterSize;
  int inVol = nPlane * inSize2;
  int outVol = nFilter * nPlane * outSize2;
  
 
  int isample = gId / outVol;
  int cubes = gId % outVol; 
  int iMap = cubes / (outSize2); //ith feature map (ith filter)
  int iPlane = iMap / nFilter;
  int pix = gId % outSize2; // pixel index for a sample
  
  int volOffset = isample * inVol;
    
  // the left-upper corner of the ith plane 
  int imageOffset = iPlane * inSize2;
  
  // calculate the left-upper corner for performing convolution
  int row_ = pix / outSize;
  int col_ = pix % outSize;
  
  // the left-upper corner for performing convolution
  int y0 = row_  * stride - padding;
  int x0 = col_  * stride - padding;
  
  // index of the left-upper corner of the filter
  int filterOffset = iMap * filterSize2;
  float sum = 0;
  int iy = 0;

  // deal with left zero pads
  if (y0 < 0) iy += padding;

  int filterSizeY = filterSize;
  int filterSizeX = filterSize;

  // deal with bottom zero pads
  if (y0 + filterSize >= inSize ) filterSizeY -= padding;
  // deal with right zero pads
  if (x0 + filterSize >= inSize ) filterSizeX -= padding;
 
  while(iy < filterSizeY){
    int ix = 0;
    int iyoffset = y0 + iy;
    int dy = iyoffset * inSize;
    int dyF =  iy * filterSize;
    // deal with right zero pads
    if (x0 < 0) ix += padding;
    while(ix < filterSizeX){
      int ixoffset = x0 + ix;
      int inx = ixoffset + dy + imageOffset + volOffset;
      int inxF = ix + dyF + filterOffset;
      sum += filter[inxF] * images[inx];
      ix++;
    }
    
    iy++;
  }
  output[gId] = sum + bias[iMap];
}



kernel void forward_conv_cl(const int nPlane,
		     const int nFilter, 
		     const int filterSize,
		     const int stride,
		     const int inSize,
		     const int outSize,
		     global const float* images,
		     global const float* filter,
		     global const float* bias,
		     global float* output) {
  
  int gId = get_global_id(0); // index of the output array
  int inSize2 = inSize * inSize;
  int outSize2 = outSize * outSize;
  int filterSize2 = filterSize * filterSize;
  int inVol = nPlane * inSize2;
  int outVol = nFilter * nPlane * outSize2;
  
 
  int isample = gId / outVol;
  int cubes = gId % outVol; 
  int iMap = cubes / (outSize2); //ith feature map (ith filter)
  int iPlane = iMap / nFilter;
  int pix = gId % outSize2; // pixel index for a sample
  
  int volOffset = isample * inVol;
    
  // the left-upper corner of the ith plane 
  int imageOffset = iPlane * inSize2;
  
  // calculate the left-upper corner for performing convolution
  int row_ = pix / outSize;
  int col_ = pix % outSize;
  
  // the left-upper corner for performing convolution
  int y0 = row_ * stride;
  int x0 = col_ * stride;
  
  // index of the left-upper corner of the filter
  int filterOffset = iMap * filterSize2;
  
  float sum = 0;
  int iy = 0;
  while(iy < filterSize){
    int ix = 0;
    int dy = (y0+iy) * inSize;
    int dyF =  iy * filterSize;
    while(ix < filterSize){
      int dx = x0 + ix;
      int inx = dx + dy + imageOffset + volOffset;
      int inxF = ix + dyF + filterOffset;
      sum += filter[inxF] * images[inx];
      ix++;
    }
    iy++;
  }

  output[gId] = sum + bias[iMap];
}
    
