////////////////////////////////////////////////////////////////////////////////
// NDVector.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  ND Vector class supporting scanline-order traversal (calling a
//  visitor function for each entry).
//
//  Note: we use a C-style, `row-major` indexing convention (rightmost index is
//  the contiguous one) just like numpy.
//
//  Elements can be accessed in two ways:
//      list of index args:    array(i0, i1, ...)
//      vector of indices:     array(NDVectorIndex<N>)
//  Visitors applied to an array can have three signatures:
//      value-only:            f(val)
//      value and index list:  f(val, i0, i1, ...)
*///////////////////////////////////////////////////////////////////////////////
#ifndef NDVector_HH
#define NDVector_HH
#include <vector>
#include <algorithm>
#include <MeshFEM/Types.hh>
#include <MeshFEM/Future.hh>
#include <MeshFEM/function_traits.hh>
#include <MeshFEM/TemplateHacks.hh>
#include <MeshFEM/GlobalBenchmark.hh>
#include "IndexUtils.hh"

#include <iostream>
#include <cassert>

template<typename T>
class NDVector {
public:
    // Empty constructor -- leaves the data initialized
    NDVector() {}

    // Initializes only the dimensions of NDVector -- leaves the data initialized
    template<typename SizeContainer>
    NDVector(const SizeContainer &sizes) {
        resize(sizes);
    }

    // Initializes only the dimensions of NDVector -- leaves the data initialized
    template<typename... Args>
    NDVector(Args... _sizes) : NDVector(std::vector<size_t>({size_t(_sizes)...})) {
        static_assert(all_integer_parameters<Args...>(), "NDVector constructor parameters must all be integers.");
    }

    void resize(const std::vector<size_t> &sizes) {
        size_t N = sizes.size();
        if (N == 0) throw std::runtime_error("NDVector must have at least one dimension");
        bool same = (m_N == N);
        for (size_t d = 0; d < N; ++d) {
            if (sizes[d] == 0) throw std::runtime_error("NDVector dimensions should be strictly positive");
            if (same) same &= (sizes[d] == m_sizes[d]);
        }
        if (same) return;

        // Initialize array storing the Dimensions
        m_N = sizes.size();
        m_sizes = sizes;

        // Initialize the array storing the elements
        m_flatSize = totalNumberOfElements();
        m_data.resize(m_flatSize);
    }

    template<typename Derived>
    void resize(const Eigen::PlainObjectBase<Derived> &sizes) {
        static_assert(std::is_same<typename Derived::Scalar, size_t>::value, "Elements of sizes must be of type size_t");
        static_assert((Derived::ColsAtCompileTime == 1) || (Derived::RowsAtCompileTime == 1), "sizes must be a vector.");
        resize(std::vector<size_t>(sizes.data(), sizes.data() + sizes.size()));
    }

    // Return the total number of elements stored in the NDVector
    size_t size() const { return m_flatSize; }

    // Round brackets are for multi-index accessing
    template<typename... Args> std::enable_if_t<all_integer_parameters<Args...>(), const T &> operator()(Args... indices) const { return m_data[flatIndex(std::array<FirstType<Args...>, sizeof...(indices)>{FirstType<Args...>(indices)...})]; }
    template<typename... Args> std::enable_if_t<all_integer_parameters<Args...>(),       T &> operator()(Args... indices)       { return m_data[flatIndex(std::array<FirstType<Args...>, sizeof...(indices)>{FirstType<Args...>(indices)...})]; }
    template<typename IndexType> const T &operator()(const std::vector<IndexType> &indices)   const { return m_data[flatIndex(indices)]; }
    template<typename IndexType>       T &operator()(const std::vector<IndexType> &indices)         { return m_data[flatIndex(indices)]; }

    // Index using an Eigen-based Nd index type.
    template<class Derived> const T &operator()(const Eigen::DenseBase<Derived> &indices) const { return m_data[flatIndex(indices)]; }
    template<class Derived>       T &operator()(const Eigen::DenseBase<Derived> &indices)       { return m_data[flatIndex(indices)]; }

    // Square brackets are for linear index accessing
    const T &operator[](size_t idx1D) const { return m_data[idx1D]; }
          T &operator[](size_t idx1D)       { return m_data[idx1D]; }

    // Direct accessor to the data
    const aligned_std_vector<T> &data() const { return m_data; }
          aligned_std_vector<T> &data()       { return m_data; }

    using EigenData = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    Eigen::Map<      EigenData> flattened()       { return Eigen::Map<      EigenData>(m_data.data(), m_data.size()); }
    Eigen::Map<const EigenData> flattened() const { return Eigen::Map<const EigenData>(m_data.data(), m_data.size()); }

    // Visit each entry in scanline order, calling either visitor(val, idx0, idx1, ...)
    // or visitor(val) depending on visitor's signature
    template<class F>
    typename std::enable_if<function_traits<F>::arity == 2, void>::type
    visit(F &&visitor) {
        for(size_t index = 0; index < m_flatSize; ++index)
            visitor(m_data[index], unflattenIndex(index) );
    }
    template<class F>
    typename std::enable_if<function_traits<F>::arity == 1, void>::type
    visit(F &&visitor) {
        for(size_t index = 0; index < m_flatSize; ++index)
            visitor(m_data[index]);
    }
    template<class F>
    typename std::enable_if<function_traits<F>::arity == 2, void>::type
    visit(F &&visitor) const {
        for(size_t index = 0; index < m_flatSize; ++index)
            visitor(m_data[index], unflattenIndex(index) );

    }
    template<class F>
    typename std::enable_if<function_traits<F>::arity == 1, void>::type
    visit(F &&visitor) const {
        for(size_t index = 0; index < m_flatSize; ++index)
            visitor(m_data[index]);
    }


    void fill(const T &value) { for(auto & m_value : m_data) m_value = value; }

    template<class DataContainer>
    std::enable_if_t<!std::is_arithmetic<DataContainer>::value> fill(const DataContainer &values) {
        if (size_t(values.size()) != m_data.size())
            throw std::runtime_error("Can not fill NDvector of size " + std::to_string(m_data.size())
                                     + " with input vector of size " + std::to_string(values.size()));
        for (size_t i = 0; i < size_t(values.size()); ++i)
            m_data[i] = values[i];
    }

    void swap(NDVector<T> &other) {
        std::swap(other.m_sizes,    this->m_sizes);
        std::swap(other.m_data,     this->m_data);
        std::swap(other.m_N,        this->m_N);
        std::swap(other.m_flatSize, this->m_flatSize);
    }

    // Compute, for each element of the array, the average difference of this element with its neighbors
    // Neighbors are the elements with a difference of 1 in one index of the multi-index notation
    // eg in 2D, element (i,j) neighbors are (i-1,j), (i+1,j), (i,j-1), (i,j+1)
    // \return NDVector of the all the average difference to neighbors
    NDVector<T> differenceToNeighborsAverage() const {
        NDVector<T> result(m_sizes);

        for (size_t ei = 0 ; ei < m_flatSize; ++ei) {
            std::vector<size_t> NDindices = unflattenIndex(ei);
            size_t nbNeighbors = 0;
            T average = 0;
            for (size_t dim = 0; dim < m_N; ++dim) {
                if (NDindices[dim] != 0) {
                    --NDindices[dim];
                    average += (*this)(NDindices);
                    ++NDindices[dim];
                    ++nbNeighbors;
                }
                if (NDindices[dim] != m_sizes[dim] - 1) {
                    ++NDindices[dim];
                    average += (*this)(NDindices);
                    --NDindices[dim];
                    ++nbNeighbors;
                }
            }
            average /= nbNeighbors;
            result[ei] = average - m_data[ei];
        }
        return result;
    }

    void visitLayer(size_t layerIndex, const std::function<void(size_t)> &callback) const {
        return visitLayer(layerIndex, m_sizes, callback);
    }

    void visitSupportingRegion(std::vector<size_t> voxelIndex, const std::function<void(size_t)> &callback) const {
        return visitSupportingRegion(voxelIndex, m_sizes, callback);
    }

    template<class SizeContainer>
    static void visitLayer(size_t layerIndex, const SizeContainer &sizes, const std::function<void(size_t)> &callback) {
        size_t N = sizes.size();
        if ((layerIndex < 0) || (layerIndex > sizes[1] - 1))
            throw std::runtime_error("NDVector has no layer " + std::to_string(layerIndex) + ".");
        if (N == 2) {
            std::vector<size_t> idx = {0, layerIndex};
            for (size_t i = 0; i < sizes[0]; i++) {
                idx[0] = i;
                callback(NDVector<T>::flatIndex(idx, sizes));
            }
        }
        else if (N == 3) {
            std::vector<size_t> idx = {0, layerIndex, 0};
            for (size_t i = 0; i < sizes[0]; i++) {
                idx[0] = i;
                for (size_t j = 0; j < sizes[2]; j++) {
                    idx[2] = j;
                    callback(NDVector<T>::flatIndex(idx, sizes));
                }
            }
        }
    }

    template<class SizeContainer>
    static void visitSupportingRegion(std::vector<size_t> voxelIndex, const SizeContainer &sizes, const std::function<void(size_t)> &callback) {
        size_t N = sizes.size();
        if (voxelIndex[1] == 0)
            throw std::runtime_error("Lower layer has no supporting region.");
        std::vector<int> supportCenter(voxelIndex.cbegin(), voxelIndex.cend()); // cast to int: indices could become negative
        supportCenter[1] -= 1;
        std::vector<int> currentIndex(supportCenter);
        callback(NDVector<T>::flatIndex(currentIndex, sizes)); // voxel below
        for(size_t d = 0; d < N-1; d++) {                      // "side" voxels
            currentIndex = supportCenter;
            currentIndex[d] -= 1;
            if (NDIndexInBound(currentIndex, sizes))
                callback(NDVector<T>::flatIndex(currentIndex, sizes));
            currentIndex[d] += 2;
            if (NDIndexInBound(currentIndex, sizes))
                callback(NDVector<T>::flatIndex(currentIndex, sizes));
        }
    }

    // Check if unflattened index `gridIndices` falls within the bounds of an NDVector of size `sizes`.
    template<class IndexContainer, class SizeContainer>
    static bool NDIndexInBound(const IndexContainer &gridIndices, const SizeContainer &sizes) {
        if (size_t(gridIndices.size()) != size_t(sizes.size()))
            throw std::runtime_error("Invalid number of indices, got " + std::to_string(sizes.size())
                                     + " indices but got " + std::to_string(gridIndices.size()) + " sizes");
        for (size_t d = 0; d < size_t(sizes.size()); d++)
            if ((gridIndices[d] < 0) || (decltype(sizes[d])(gridIndices[d]) >= sizes[d]))
                return false;
        return true;
    }

    template<class IndexContainer>
    bool NDIndexInBound(const IndexContainer &gridIndices) const { return NDIndexInBound(gridIndices, sizes()); }

    // Get the linear index for an NDVector with dimensions `sizes`
    // corresponding to the input multi-indices stored in `indices`.
    // Template parameters IndexContainer and SizeContainer are, e.g., std::vector<size_t> or std::array<size_t, N>.
    template<bool checked = true, class IndexContainer, class SizeContainer>
    static size_t flatIndex(const IndexContainer &indices, const SizeContainer &sizes) {
        if (checked) {
            if (!(NDIndexInBound(indices, sizes)))
                throw std::runtime_error("Indices out of bounds of NDVector");
        }

        // (i)     -> i
        // (i,j)   -> Ny * i + j
        // (i,j,k) -> Nz * (Ny * i + j) + k
        const size_t N = indices.size();
        size_t result = indices[0];
        for (size_t d = 1; d < N; ++d)
            result = result * sizes[d] + indices[d];

        return result;
    }

    template<class IndexContainer, class SizeContainer>
    static auto flatIndexConstexpr(const IndexContainer &indices, const SizeContainer &sizes) {
        constexpr size_t N = IndexContainer::RowsAtCompileTime;
        static_assert(SizeContainer::RowsAtCompileTime == N, "Dimension mismatch");
        static_assert((N > 0) && (N <= 3),                   "Unimplemented dimension");
        if (N == 1) return indices[0];
        if (N == 2) return sizes[1] * indices[0] + indices[1];
        return sizes[2] * (sizes[1] * indices[0] + indices[1]) + indices[2];
    }

    // Get the linear index corresponding to the input multi-indices stored in a std::vector (in a NDVector of the same
    // dimensions as *this)
    template<bool checked = true, class IndexContainer>
    size_t flatIndex(const IndexContainer &indices) const {
        size_t result = flatIndex<checked>(indices, m_sizes);
        if (checked && (result >= m_flatSize))
            throw std::runtime_error("Index out of range");
        return result;
    }

    // Get the multi-index of the data corresponding to the input linear index (in a NDVector of the same dimensions as *this)
    std::vector<size_t> unflattenIndex (size_t flatIndex) const {
        return unflattenIndex(flatIndex, m_sizes);
    }

    template<size_t N>
    Eigen::Array<size_t, N, 1> unflattenIndex(size_t flatIndex) const {
        Eigen::Array<size_t, N, 1> result;
        unflattenIndex(flatIndex, m_sizes, result);
        return result;
    }

    template<class SizeContainer, class ResultContainer>
    static void unflattenIndex(size_t flatIndex, const SizeContainer &sizes, ResultContainer &result) {
        const size_t N = sizes.size();
        result.resize(N);
        for (size_t d = N - 1; d < N; --d) {
            size_t s = sizes[d];
            result[d] = flatIndex % s;
            flatIndex /= s;
        }

        if (flatIndex != 0)
            throw std::runtime_error("Index out of range");

    }

    template<class SizeContainer, class ResultContainer>
    static void unflattenIndexConstexpr(typename SizeContainer::Scalar flatIndex,
                                        const SizeContainer &sizes, ResultContainer &result) {
        constexpr size_t N = SizeContainer::RowsAtCompileTime;
        static_assert(ResultContainer::RowsAtCompileTime == N, "Dimension mismatch");
        if (N == 1) { result[0] =  flatIndex; return; }
        if (N == 2) { result[1] = (flatIndex % sizes[1]); result[0] = flatIndex / sizes[1]; return; }
        if (N == 3) { result[2] = (flatIndex % sizes[2]); flatIndex /= sizes[2];
                      result[1] = (flatIndex % sizes[1]); result[0] = flatIndex / sizes[1]; return; }
    }

    // Get the multi-index of the data corresponding to the input linear index, in a NDVector of dimensions given by
    // input "sizes"
    // SizeContainer is, e.g., std::vector<size_t>
    template<class SizeContainer>
    static std::vector<size_t> unflattenIndex(size_t flatIndex, const SizeContainer &sizes) {
        std::vector<size_t> result;
        unflattenIndex(flatIndex, sizes, result);
        return result;
    }

    // Returns: the number of elements in each dimension
    const std::vector<size_t> &sizes() const { return m_sizes; }

    // Increment in flat index induced by changing an ND-index
    template<class IndexContainer>
    void getFlatIndexIncrements(IndexContainer &result) const {
        getFlatIndexIncrements(m_sizes, result);
    }

    // Increment in flat index induced by changing an ND-index
    template<class IndexContainer, class IndexContainer2>
    static void getFlatIndexIncrements(const IndexContainer2 &sizes, IndexContainer &result) {
        const size_t N = sizes.size();
        result.resize(N);
        result[N - 1] = 1;
        for (int d = N - 2; d >= 0; --d)
            result[d] = result[d + 1] * sizes[d + 1];
    }

    // Iterators
    typename aligned_std_vector<T>::iterator begin() { return m_data.begin(); }
    typename aligned_std_vector<T>::iterator   end() { return m_data.end(); }
    typename aligned_std_vector<T>::const_iterator begin() const { return m_data.begin(); }
    typename aligned_std_vector<T>::const_iterator   end() const { return m_data.end(); }

private:
    // Get the linear index corresponding to the input multi-indices (in a NDVector of the same dimensions as *this)
    template<typename... Args> size_t flatIndex(Args... Indices) const {
        if (sizeof...(Indices) != m_N)
            throw std::runtime_error("Invalid number of indices, expected "
                                     + std::to_string(m_N)
                                     + " but got " + std::to_string(sizeof...(Indices)));
        std::vector<size_t> indices{{size_t(Indices)...}};
        return flatIndex(indices);
    }

    // Return the total number of elements that can be stored in m_data
    size_t totalNumberOfElements() const {
        size_t nbElements = 1;
        for (auto const & dim : m_sizes) {
            nbElements *= dim;
        }
        return nbElements;
    }

    // The flattened, 1D size of the ND vector (total number of entries).
    size_t m_flatSize;

    // number of dimensions: "N" of NDVector
    size_t m_N = 0;
    // number of elements to store in each dimension
    std::vector<size_t> m_sizes;

    // The actual storage member
    aligned_std_vector<T> m_data;
};

#endif /* end of include guard: NDVector_HH */
