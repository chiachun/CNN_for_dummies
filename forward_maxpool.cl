kernel void forward_maxpool(const int nPlane,
			    const int filterSize,
			    const int stride,
			    const int padding,
			    const int inSize,
			    const int outSize,
			    global const float* images,
			    global float* output,
			    global int* selected){
  int gId = get_global_id(0);
  int inSize2 = inSize * inSize;
  int outSize2 = outSize * outSize;
  int inVol = nPlane * inSize2;
  int outVol = nPlane * outSize2;

  int isample = gId / outVol;
  int cubes = gId % outVol; 
  int iMap = cubes / (outSize2); //ith feature map (ith filter)
  int pix = gId % outSize2; // pixel index for a sample
  
  int volOffset = isample * inVol;
    
  // the left-upper corner of the ith plane 
  int imageOffset = iMap * inSize2;
  
  // calculate the left-upper corner for performing convolution
  int row_ = pix / outSize;
  int col_ = pix % outSize;
  
  // the left-upper corner for performing convolution
  int y0 = row_  * stride - padding;
  int x0 = col_  * stride - padding;
  
  int iy = 0;
  // deal with left zero pads
  if (y0 < 0) iy += padding;

  int filterSizeY = filterSize;
  int filterSizeX = filterSize;

  // deal with bottom zero pads
  if (y0 + filterSize >= inSize ) filterSizeY -= padding;
  // deal with right zero pads
  if (x0 + filterSize >= inSize ) filterSizeX -= padding;
  
  int lmax = 0;
  int imax = 0;
  while(iy < filterSizeY){
    int ix = 0;
    int iyoffset = y0 + iy;
    int dy = iyoffset * inSize;
    // deal with right zero pads
    if (x0 < 0) ix += padding;
    while(ix < filterSizeX){
      int ixoffset = x0 + ix;
      int inx = ixoffset + dy + imageOffset + volOffset;
      if (images[inx]> lmax) {lmax = images[inx]; imax = inx;}
      ix++;
    }
    
    iy++;
  }
  if (imax == 0) imax = x0 + y0 + imageOffset + volOffset;
  output[gId] = lmax;
  selected[gId] = imax;
}
