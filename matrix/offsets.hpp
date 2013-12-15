
#ifndef __MATRIX_OFFSETS_H
#define __MATRIX_OFFSETS_H

#include <vector>
#include <array>

/*
 * I want this to work in CUDA kernels,
 * so no C++11 :(
 */
template <int NDims,
	  typename TDim=int,
	  typename Storage=std::array<TDim,NDims>>
class Offsets
{

private:

    Storage dim;

public:

  const static TDim N = NDims;

  /*
   * Constructors 
   */
  Offsets()
  {
    for(int i=0; i<N; i++)
      dim[i] = 0;
  }

  template<typename T>
  Offsets(T dims)
  {
    for(int i=0; i<N; i++)
      this->dim[i] = dims[i];
  }

  void set_dim(TDim i, TDim d)
  {
    dim[i] = d;
  }

  TDim get_dim(TDim i)
  {
    return dim[i];
  }
  
  TDim operator[](TDim i)
  {
    return dim[i];
  }

};







#endif
