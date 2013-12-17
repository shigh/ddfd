/*
 * Tinkering with methods required to develop n-dimensional
 * data structures
 */

#ifndef __MATRIX_OFFSETS_H
#define __MATRIX_OFFSETS_H

#include <vector>
#include <array>

/*
 * Helper class for Offsets, used as a proxy
 * for chained [] operators in Offsets
 */
template<typename T>
class ProxyOffsets
{
private:

  T& _base;

  int _curr;
  int _offset;

public:

  /*
   * First call from Offsets
   */
  ProxyOffsets(T& base, int i): _base(base)
  {
    _offset = i*_base.offs[0];
    _curr   = 1;
  }

  /*
   * Chain the [] operators
   */
  ProxyOffsets<T>& operator[](int i)
  {
    _offset += i*_base.offs[_curr++];
    return *this;
  }

  /*
   * Convert to integer offset
   */
  operator int() { return _offset; }
  
};


/*
 * Calculate the linear array offset in ndimensions.
 *
 * This is mostly for me to learn
 *
 * TODO: Add timing test
 * 
 * I want this to work in CUDA kernels,
 * so no C++11 :(
 */
template <int NDims,
	  typename Storage=std::array<int, NDims>>
class Offsets
{
  template <typename T> friend class ProxyOffsets;
private:

  /*
   * The numeric size of each dimension
   */
  Storage dim;

  /*
   * The term multiplied by the element number to
   * get the total offset
   *
   * offset[z][y][x] = z*offs[0] + y*offs[1] + x*offs[0]
   */
  Storage offs;

  void calc_offs();
  
public:

  const static int N = NDims;

  typedef ProxyOffsets< Offsets<NDims, Storage> > ProxyType;

  /*
   * Constructors 
   */
  Offsets()
  {
    for(int i=0; i<N; i++)
      dim[i] = 0;
    calc_offs();
  }

  template<typename T>
  Offsets(T dims)
  {
    for(int i=0; i<N; i++)
      this->dim[i] = dims[i];
    calc_offs();
  }

  void set_dim(int i, int d) { dim[i] = d; }

  int get_dim(int i) { return dim[i]; }

  ProxyType operator[](int z) { return ProxyType(*this, z); }

};

template <int NDims, typename Storage>
void Offsets<NDims, Storage>::calc_offs()
{
  int j = 1;
  offs[N-1] = 1;
  for(int i=N-2; i>=0; i--)
    {
      j = j*dim[i+1];
      offs[i] = j;
    }
}


#endif
