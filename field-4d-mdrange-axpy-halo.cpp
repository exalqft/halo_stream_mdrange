/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// ************************************************************************
//
// Rephrasing as a benchmark for linear algebra by Bartosz Kostrzewa, Aniket Sen (Uni Bonn) 
//
//@HEADER
*/

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <getopt.h>
#include <utility>
#include <iostream>
#include <limits>

#include <sys/time.h>


#define STREAM_NTIMES 20

using val_t = Kokkos::complex<double>;

constexpr val_t ainit(1.0, 0.1);
constexpr val_t binit(1.1, 0.2);
constexpr val_t cinit(1.3, 0.3);
// constexpr val_t dinit(1.4, 0.4);

#define HLINE "-------------------------------------------------------------\n"

using Field =
    Kokkos::View<val_t****, Kokkos::MemoryTraits<Kokkos::Restrict>>;

#if defined(KOKKOS_ENABLE_CUDA)
using constField =
    Kokkos::View<const val_t ****, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
#else
using constField =
    Kokkos::View<const val_t ****, Kokkos::MemoryTraits<Kokkos::Restrict>>;
#endif

using StreamIndex = int;

template <int rank>
using Policy      = Kokkos::MDRangePolicy<Kokkos::Rank<rank>>;

template <std::size_t... Idcs>
constexpr Kokkos::Array<std::size_t, sizeof...(Idcs)>
make_repeated_sequence_impl(std::size_t value, std::integer_sequence<std::size_t, Idcs...>)
{
  return { ((void)Idcs, value)... };
}

template <std::size_t N>
constexpr Kokkos::Array<std::size_t,N> 
make_repeated_sequence(std::size_t value)
{
  return make_repeated_sequence_impl(value, std::make_index_sequence<N>{});
}

template <typename V>
auto
get_tiling(const V view)
{
  constexpr auto rank = view.rank_dynamic();
  // extract the dimensions from the view layout (assuming no striding)
  const auto & dimensions = view.layout().dimension;
  Kokkos::Array<std::size_t,rank> dims;
  for(int i = 0; i < rank; ++i){
    dims[i] = dimensions[i];
  }
  // extract the recommended tiling for this view from a "default" policy 
  const auto rec_tiling = Policy<rank>(make_repeated_sequence<rank>(0),dims).tile_size_recommended();
  
  if constexpr (std::is_same_v<typename V::execution_space, Kokkos::DefaultHostExecutionSpace>){
    // for OpenMP we parallelise over the two outermost (leftmost) dimensions and so the chunk size
    // for the innermost dimensions corresponds to the view extents
    return Kokkos::Array<std::size_t,rank>({1,1,view.extent(2),view.extent(3)});
  } else {
    // for GPUs we use the recommended tiling for now, we just need to convert it appropriately
    // from "array_index_type"
    // unfortunately the recommended tile size may exceed the maximum block size on GPUs 
    // for large ranks -> let's cap the tiling at 4 dims
    constexpr auto max_rank = rank > 4 ? 4 : rank;
    Kokkos::Array<std::size_t,max_rank> res;
    for(int i = 0; i < max_rank; ++i){
      res[i] = rec_tiling[i];
    }
    return res;
  }
}

struct deviceField {
  deviceField() = delete;

  deviceField(std::size_t N0, std::size_t N1, std::size_t N2, std::size_t N3, const val_t init)
  {
    do_init(N0,N1,N2,N3,view,init);
  }
  
  // need to take care of 'this'-pointer capture 
  void
  do_init(std::size_t N0, std::size_t N1, std::size_t N2, std::size_t N3, 
          Field & V, const val_t init){
    Kokkos::realloc(Kokkos::WithoutInitializing, V, N0, N1, N2, N3);
    
    // need a const view to get the constexpr rank
    const Field vconst = V;
    constexpr auto rank = vconst.rank_dynamic();
    const auto tiling = get_tiling(vconst);
    
    Kokkos::parallel_for(
      "init", 
      Policy<rank>(make_repeated_sequence<rank>(0), {N0,N1,N2,N3}, tiling),
      KOKKOS_LAMBDA(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l)
      {
        V(i,j,k,l) = init;
      }
    );
    Kokkos::fence();
  }

  KOKKOS_FORCEINLINE_FUNCTION val_t & operator()(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l) const {
    return view(i,j,k,l);
  }

  KOKKOS_FORCEINLINE_FUNCTION val_t & operator()(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l) {
    return view(i,j,k,l);
  }

  Field view;
};

struct deviceField_h {
  deviceField_h() = delete;

  deviceField_h(std::size_t N0, std::size_t N1, std::size_t N2, std::size_t N3, const val_t init,
              const Kokkos::Array<int,4> dims_partitioned = {0,0,0,0}) 
              : dims({N0,N1,N2,N3}), halo_offsets(dims_partitioned)  
  {
    do_init(view,init);
  }
  
  // need to take care of 'this'-pointer capture 
  void
  do_init(Field & V, const val_t init){
    Kokkos::Array<std::size_t,4> extents;
    for(int i = 0; i < 4; ++i) {
      extents[i] = dims[i] + 2*halo_offsets[i];
    }
    Kokkos::realloc(Kokkos::WithoutInitializing, V, extents[0], extents[1], extents[2], extents[3]);
    
    // need a const view to get the constexpr rank
    const Field vconst = V;
    constexpr auto rank = vconst.rank_dynamic();
    const auto tiling = get_tiling(vconst);
    
    Kokkos::parallel_for(
      "init", 
      Policy<rank>(make_repeated_sequence<rank>(0), dims, tiling),
      KOKKOS_CLASS_LAMBDA(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l)
      {
        const StreamIndex ii = i + this->halo_offsets[0];
        const StreamIndex jj = j + this->halo_offsets[1];
        const StreamIndex kk = k + this->halo_offsets[2];
        const StreamIndex ll = l + this->halo_offsets[3];
        V(ii,jj,kk,ll) = init;
      }
    );
    Kokkos::fence();
  }

  KOKKOS_FORCEINLINE_FUNCTION val_t & operator()(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l) const {
    const StreamIndex ii = i + halo_offsets[0];
    const StreamIndex jj = j + halo_offsets[1];
    const StreamIndex kk = k + halo_offsets[2];
    const StreamIndex ll = l + halo_offsets[3];
    return view(ii,jj,kk,ll);
  }

  KOKKOS_FORCEINLINE_FUNCTION val_t & operator()(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l) {
    const StreamIndex ii = i + halo_offsets[0];
    const StreamIndex jj = j + halo_offsets[1];
    const StreamIndex kk = k + halo_offsets[2];
    const StreamIndex ll = l + halo_offsets[3];
    return view(ii,jj,kk,ll);
  }

  Field view;
  const Kokkos::Array<std::size_t,4> dims;
  const Kokkos::Array<int,4> halo_offsets;
};

int parse_args(int argc, char **argv, StreamIndex &stream_array_size) {
  // Defaults
  stream_array_size = 32;

  const std::string help_string =
      "  -n <N>, --nelements <N>\n"
      "     Create stream views containing <N>^4 elements.\n"
      "     Default: 32\n"
      "  -h, --help\n"
      "     Prints this message.\n"
      "     Hint: use --kokkos-help to see command line options provided by "
      "Kokkos.\n";

  static struct option long_options[] = {
      {"nelements", required_argument, NULL, 'n'},
      {"help", no_argument, NULL, 'h'},
      {NULL, 0, NULL, 0}};

  int c;
  int option_index = 0;
  while ((c = getopt_long(argc, argv, "n:h", long_options, &option_index)) !=
         -1)
    switch (c) {
      case 'n': stream_array_size = atoi(optarg); break;
      case 'h':
        printf("%s", help_string.c_str());
        return -2;
        break;
      case 0: break;
      default:
        printf("%s", help_string.c_str());
        return -1;
        break;
    }
  return 0;
}

void axpy(const val_t a, const deviceField x, const deviceField y) {
  constexpr auto rank = x.view.rank_dynamic();
  const auto stream_array_size = x.view.extent(0);
  const auto tiling = get_tiling(x.view);
  Kokkos::parallel_for(
      "axpy", 
      Policy<rank>(make_repeated_sequence<rank>(0), {stream_array_size,stream_array_size,stream_array_size,stream_array_size}, tiling),
      KOKKOS_LAMBDA(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l)
      {
        y(i,j,k,l) += a * x(i,j,k,l);
      });
  Kokkos::fence();
}

void axpy_h(const val_t a, const deviceField x, const deviceField_h y) {
  constexpr auto rank = x.view.rank_dynamic();
  const auto stream_array_size = x.view.extent(0);
  const auto tiling = get_tiling(x.view);
  Kokkos::parallel_for(
      "axpy", 
      Policy<rank>(make_repeated_sequence<rank>(0), {stream_array_size,stream_array_size,stream_array_size,stream_array_size}, tiling),
      KOKKOS_LAMBDA(const StreamIndex i, const StreamIndex j, const StreamIndex k, const StreamIndex l)
      {
        y(i,j,k,l) += a * x(i,j,k,l);
      });
  Kokkos::fence();
}

int run_benchmark(const StreamIndex stream_array_size) {
  printf("Reports fastest timing per kernel\n");
  printf("Creating Views...\n");
  
  const double nelem = (double)stream_array_size*
                       (double)stream_array_size*
                       (double)stream_array_size*
                       (double)stream_array_size;

  printf("Memory Sizes:\n");
  printf("- Array Size:  %" PRIu64 "^4\n",
         static_cast<uint64_t>(stream_array_size));
  printf("- Size per field:                 %12.2f MB\n",
         1.0e-6 * nelem * (double)sizeof(val_t));

  printf("Benchmark kernels will be performed for %d iterations.\n",
         STREAM_NTIMES);

  printf(HLINE);

  double axpy_Time  = std::numeric_limits<double>::max();
  double axpy_h_no_offset_Time      = std::numeric_limits<double>::max();
  double axpy_h_offset_x_Time       = std::numeric_limits<double>::max();
  double axpy_h_offset_y_Time       = std::numeric_limits<double>::max();
  double axpy_h_offset_z_Time       = std::numeric_limits<double>::max();
  double axpy_h_offset_t_Time       = std::numeric_limits<double>::max();
  double axpy_h_offset_xy_Time      = std::numeric_limits<double>::max();
  double axpy_h_offset_xz_Time      = std::numeric_limits<double>::max();
  double axpy_h_offset_xt_Time      = std::numeric_limits<double>::max();
  double axpy_h_offset_yz_Time      = std::numeric_limits<double>::max();
  double axpy_h_offset_yt_Time      = std::numeric_limits<double>::max();
  double axpy_h_offset_zt_Time      = std::numeric_limits<double>::max();
  double axpy_h_offset_xyz_Time     = std::numeric_limits<double>::max();
  double axpy_h_offset_xyt_Time     = std::numeric_limits<double>::max();
  double axpy_h_offset_xzt_Time     = std::numeric_limits<double>::max();
  double axpy_h_offset_yzt_Time     = std::numeric_limits<double>::max();
  double axpy_h_offset_xyzt_Time    = std::numeric_limits<double>::max();

  printf("Starting benchmarking...\n");

  printf(HLINE);

  int rc = 0;

  Kokkos::Timer timer;

  deviceField x(stream_array_size, stream_array_size, stream_array_size, stream_array_size, ainit);

  {
  deviceField y(stream_array_size, stream_array_size, stream_array_size, stream_array_size, binit);
  
  for(StreamIndex k = 0; k < STREAM_NTIMES; ++k) {
    timer.reset();
    axpy(cinit, x, y);
    axpy_Time = std::min(axpy_Time, timer.seconds());
  }

  printf("axpy                     %11.4f GB/s\n",
         1.0e-9 * 2.0 * nelem * sizeof(val_t) / axpy_Time);
  }

  {
  deviceField_h y_h_no_offset(stream_array_size, stream_array_size, stream_array_size, stream_array_size, binit, {0,0,0,0});

  for(StreamIndex k = 0; k < STREAM_NTIMES; ++k) {
    timer.reset();
    axpy_h(cinit, x, y_h_no_offset);
    axpy_h_no_offset_Time = std::min(axpy_h_no_offset_Time, timer.seconds());
  }

  printf("axpy_h_no_offset         %11.4f GB/s\n",
         1.0e-9 * 2.0 * nelem * sizeof(val_t) / axpy_h_no_offset_Time);
  
  }

  {
  deviceField_h y_h_offset_x(stream_array_size, stream_array_size, stream_array_size, stream_array_size, binit, {1,0,0,0});

  for(StreamIndex k = 0; k < STREAM_NTIMES; ++k) {
    timer.reset();
    axpy_h(cinit, x, y_h_offset_x);
    axpy_h_offset_x_Time = std::min(axpy_h_offset_x_Time, timer.seconds());
  }

  printf("axpy_h_offset_x          %11.4f GB/s\n",
         1.0e-9 * 2.0 * nelem * sizeof(val_t) / axpy_h_offset_x_Time);
  }

  {
  deviceField_h y_h_offset_y(stream_array_size, stream_array_size, stream_array_size, stream_array_size, binit, {0,1,0,0});

  for(StreamIndex k = 0; k < STREAM_NTIMES; ++k) {
    timer.reset();
    axpy_h(cinit, x, y_h_offset_y);
    axpy_h_offset_y_Time = std::min(axpy_h_offset_y_Time, timer.seconds());
  }

  printf("axpy_h_offset_y          %11.4f GB/s\n",
         1.0e-9 * 2.0 * nelem * sizeof(val_t) / axpy_h_offset_y_Time);
  }

  {
  deviceField_h y_h_offset_z(stream_array_size, stream_array_size, stream_array_size, stream_array_size, binit, {0,0,1,0});

  for(StreamIndex k = 0; k < STREAM_NTIMES; ++k) {
    timer.reset();
    axpy_h(cinit, x, y_h_offset_z);
    axpy_h_offset_z_Time = std::min(axpy_h_offset_z_Time, timer.seconds());
  }

  printf("axpy_h_offset_z          %11.4f GB/s\n",
         1.0e-9 * 2.0 * nelem * sizeof(val_t) / axpy_h_offset_z_Time);
  }

  {
  deviceField_h y_h_offset_t(stream_array_size, stream_array_size, stream_array_size, stream_array_size, binit, {0,0,0,1});

  for(StreamIndex k = 0; k < STREAM_NTIMES; ++k) {
    timer.reset();
    axpy_h(cinit, x, y_h_offset_t);
    axpy_h_offset_t_Time = std::min(axpy_h_offset_t_Time, timer.seconds());
  }

  printf("axpy_h_offset_t          %11.4f GB/s\n",
         1.0e-9 * 2.0 * nelem * sizeof(val_t) / axpy_h_offset_t_Time);
  }

  {
  deviceField_h y_h_offset_xy(stream_array_size, stream_array_size, stream_array_size, stream_array_size, binit, {1,1,0,0});

  for(StreamIndex k = 0; k < STREAM_NTIMES; ++k) {
    timer.reset();
    axpy_h(cinit, x, y_h_offset_xy);
    axpy_h_offset_xy_Time = std::min(axpy_h_offset_xy_Time, timer.seconds());
  }

  printf("axpy_h_offset_xy         %11.4f GB/s\n",
         1.0e-9 * 2.0 * nelem * sizeof(val_t) / axpy_h_offset_xy_Time);
  }

  {
  deviceField_h y_h_offset_xz(stream_array_size, stream_array_size, stream_array_size, stream_array_size, binit, {1,0,1,0});

  for(StreamIndex k = 0; k < STREAM_NTIMES; ++k) {
    timer.reset();
    axpy_h(cinit, x, y_h_offset_xz);
    axpy_h_offset_xz_Time = std::min(axpy_h_offset_xz_Time, timer.seconds());
  }

  printf("axpy_h_offset_xz         %11.4f GB/s\n",
         1.0e-9 * 2.0 * nelem * sizeof(val_t) / axpy_h_offset_xz_Time);
  }

  {
  deviceField_h y_h_offset_xt(stream_array_size, stream_array_size, stream_array_size, stream_array_size, binit, {1,0,0,1});

  for(StreamIndex k = 0; k < STREAM_NTIMES; ++k) {
    timer.reset();
    axpy_h(cinit, x, y_h_offset_xt);
    axpy_h_offset_xt_Time = std::min(axpy_h_offset_xt_Time, timer.seconds());
  }

  printf("axpy_h_offset_xt         %11.4f GB/s\n",
         1.0e-9 * 2.0 * nelem * sizeof(val_t) / axpy_h_offset_xt_Time);
  }

  {
  deviceField_h y_h_offset_yz(stream_array_size, stream_array_size, stream_array_size, stream_array_size, binit, {0,1,1,0});

  for(StreamIndex k = 0; k < STREAM_NTIMES; ++k) {
    timer.reset();
    axpy_h(cinit, x, y_h_offset_yz);
    axpy_h_offset_yz_Time = std::min(axpy_h_offset_yz_Time, timer.seconds());
  }

  printf("axpy_h_offset_yz         %11.4f GB/s\n",
         1.0e-9 * 2.0 * nelem * sizeof(val_t) / axpy_h_offset_yz_Time);
  }

  {
  deviceField_h y_h_offset_yt(stream_array_size, stream_array_size, stream_array_size, stream_array_size, binit, {0,1,0,1});

  for(StreamIndex k = 0; k < STREAM_NTIMES; ++k) {
    timer.reset();
    axpy_h(cinit, x, y_h_offset_yt);
    axpy_h_offset_yt_Time = std::min(axpy_h_offset_yt_Time, timer.seconds());
  }

  printf("axpy_h_offset_yt         %11.4f GB/s\n",
         1.0e-9 * 2.0 * nelem * sizeof(val_t) / axpy_h_offset_yt_Time);
  }

  {
  deviceField_h y_h_offset_zt(stream_array_size, stream_array_size, stream_array_size, stream_array_size, binit, {0,0,1,1});

  for(StreamIndex k = 0; k < STREAM_NTIMES; ++k) {
    timer.reset();
    axpy_h(cinit, x, y_h_offset_zt);
    axpy_h_offset_zt_Time = std::min(axpy_h_offset_zt_Time, timer.seconds());
  }

  printf("axpy_h_offset_zt         %11.4f GB/s\n",
         1.0e-9 * 2.0 * nelem * sizeof(val_t) / axpy_h_offset_zt_Time);
  }

  {
  deviceField_h y_h_offset_xyz(stream_array_size, stream_array_size, stream_array_size, stream_array_size, binit, {1,1,1,0});

  for(StreamIndex k = 0; k < STREAM_NTIMES; ++k) {
    timer.reset();
    axpy_h(cinit, x, y_h_offset_xyz);
    axpy_h_offset_xyz_Time = std::min(axpy_h_offset_xyz_Time, timer.seconds());
  }

  printf("axpy_h_offset_xyz        %11.4f GB/s\n",
         1.0e-9 * 2.0 * nelem * sizeof(val_t) / axpy_h_offset_xyz_Time);

  }

  {
  deviceField_h y_h_offset_xyt(stream_array_size, stream_array_size, stream_array_size, stream_array_size, binit, {1,1,0,1});

  for(StreamIndex k = 0; k < STREAM_NTIMES; ++k) {
    timer.reset();
    axpy_h(cinit, x, y_h_offset_xyt);
    axpy_h_offset_xyt_Time = std::min(axpy_h_offset_xyt_Time, timer.seconds());
  }

  printf("axpy_h_offset_xyt        %11.4f GB/s\n",
         1.0e-9 * 2.0 * nelem * sizeof(val_t) / axpy_h_offset_xyt_Time);
  }

  {
  deviceField_h y_h_offset_xzt(stream_array_size, stream_array_size, stream_array_size, stream_array_size, binit, {1,0,1,1});

  for(StreamIndex k = 0; k < STREAM_NTIMES; ++k) {
    timer.reset();
    axpy_h(cinit, x, y_h_offset_xzt);
    axpy_h_offset_xzt_Time = std::min(axpy_h_offset_xzt_Time, timer.seconds());
  }

  printf("axpy_h_offset_xzt        %11.4f GB/s\n",
         1.0e-9 * 2.0 * nelem * sizeof(val_t) / axpy_h_offset_xzt_Time);
  }

  {
  deviceField_h y_h_offset_yzt(stream_array_size, stream_array_size, stream_array_size, stream_array_size, binit, {0,1,1,1});

  for(StreamIndex k = 0; k < STREAM_NTIMES; ++k) {
    timer.reset();
    axpy_h(cinit, x, y_h_offset_yzt);
    axpy_h_offset_yzt_Time = std::min(axpy_h_offset_yzt_Time, timer.seconds());
  }

  printf("axpy_h_offset_yzt        %11.4f GB/s\n",
         1.0e-9 * 2.0 * nelem * sizeof(val_t) / axpy_h_offset_yzt_Time);
  }

  {
  deviceField_h y_h_offset_xyzt(stream_array_size, stream_array_size, stream_array_size, stream_array_size, binit, {1,1,1,1});

  for(StreamIndex k = 0; k < STREAM_NTIMES; ++k) {
    timer.reset();
    axpy_h(cinit, x, y_h_offset_xyzt);
    axpy_h_offset_xyzt_Time = std::min(axpy_h_offset_xyzt_Time, timer.seconds());
  }

  printf("axpy_h_offset_xyzt       %11.4f GB/s\n",
         1.0e-9 * 2.0 * nelem * sizeof(val_t) / axpy_h_offset_xyzt_Time);
  }

  printf(HLINE);

  return rc;
}

int main(int argc, char *argv[]) {
  printf(HLINE);
  printf("Kokkos 4D Field MDRangePolicy Halo axpy Benchmark\n");
  printf(HLINE);

  Kokkos::initialize(argc, argv);
  int rc;
  StreamIndex stream_array_size;
  rc = parse_args(argc, argv, stream_array_size);
  if (rc == 0) {
    rc = run_benchmark(stream_array_size);
  } else if (rc == -2) {
    // Don't return error code when called with "-h"
    rc = 0;
  }
  Kokkos::finalize();

  return rc;
}