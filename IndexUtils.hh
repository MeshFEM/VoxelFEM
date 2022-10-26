////////////////////////////////////////////////////////////////////////////////
// IndexUtils.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Convenience utilities for indexing into/iterating over N-dimensional grids.
//  Two iteration modes are supported: a range-based for, and a visitor pattern
//  that has dimension-specific specializations for higher performance.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  02/10/2022 12:57:02
////////////////////////////////////////////////////////////////////////////////
#ifndef INDEXUTILS_HH
#define INDEXUTILS_HH
#include <cstdlib>
#include <MeshFEM/function_traits.hh>
#include <MeshFEM/Parallelism.hh>

template<size_t N, bool Parallel = false>
struct IndexRangeVisitor;

// Support for looping over the Nd range [begin[0], end[0]) x ... [begin[N - 1], end[N - 1])
template<class IndexContainer>
struct IndexRange {
    static constexpr size_t N = IndexContainer::RowsAtCompileTime;

    struct Iterator {
        Iterator(IndexContainer i,
                 const IndexContainer &begin,
                 const IndexContainer &end)
            : m_i(i), m_begin(begin), m_end(end) { }

        const Iterator &operator++() {
            // Update Nd index
            for (size_t d = 0; d < N; ++d) {
                ++m_i[d];
                if (m_i[d] < m_end[d]) break;
                if (d == N - 1) {
                    // The most significant index has wrapped
                    m_i = m_end;
                    return *this;
                }
                m_i[d] = m_begin[d];
            }
            return *this;
        }

        const IndexContainer &operator*() const { return m_i; }

        bool operator==(const Iterator &it) const { for (size_t d = 0; d < N; ++d) { if (it.m_i[d] != m_i[d]) return false; } return true; }
        bool operator!=(const Iterator &it) const { return !(*this == it); }

        private:
            IndexContainer m_i;
            const IndexContainer &m_begin, &m_end;
    };

    IndexRange(IndexContainer begin, IndexContainer end)
        : m_begin(begin), m_end(end) { }

    Iterator begin() const { return Iterator(m_begin, m_begin, m_end); }
    Iterator end  () const { return Iterator(  m_end, m_begin, m_end); }

    IndexContainer &beginIndex() { return m_begin; }
    IndexContainer &  endIndex() { return m_end; }

    IndexContainer sizes() const { return m_end - m_begin; }
    size_t size() const { return sizes().prod(); }

    template<bool Parallel = false, class F>
    void visit(const F &f) const {
        IndexRangeVisitor<N, Parallel>::run(f, m_begin, m_end);
    }

private:
    IndexContainer m_begin, m_end;
};

template<class IndexContainer>
static IndexRange<IndexContainer> make_index_range(IndexContainer begin, IndexContainer end) {
    return IndexRange<IndexContainer>(begin, end);
}

#include <MeshFEM/ParallelAssembly.hh>
template<size_t N, typename ThreadLocalData>
struct IndexRangeVisitorThreadLocal {
    template<class F, class DataConstructor, class IndexContainer>
    static SumLocalData<ThreadLocalData> run(const F &f, const DataConstructor &construct, const IndexContainer &begin, const IndexContainer &end) {
        SumLocalData<ThreadLocalData> localResults;
        // Parallelism is applied at outermost level
        tbb::parallel_for(tbb::blocked_range<size_t>(begin[0], end[0]), [&](const tbb::blocked_range<size_t> &r) {
            auto &data = localResults.local();
            if (!data.constructed) { construct(data.v); data.constructed = true; }
            IndexContainer i;
            for (i[0] = r.begin(); i[0] < r.end(); ++i[0])
                IndexRangeVisitor<N - 1, false>::template m_run<1>([&](const IndexContainer &iND) { f(iND, data.v); }, begin, end, i);
        });
        return localResults;
    }
};

template<size_t N, bool Parallel>
struct IndexRangeVisitor {
    // Iterate over the N-dimensional index range [begin, end) using `N` nested
    // for loops. The outermost for loop will be parallelized if `Parallel == true`.
    template<class F, class IndexContainer>
    static void run(const F &f, const IndexContainer &begin, const IndexContainer &end) {
        if (Parallel) { // Parallelism is applied at outermost level
            parallel_for_range(begin[0], end[0], [&](size_t outer_i) {
                IndexContainer i;
                i[0] = outer_i;
                IndexRangeVisitor<N - 1>::template m_run<1>(f, begin, end, i);
            });
        }
        else {
            IndexContainer i;
            for (i[0] = begin[0]; i[0] < end[0]; ++i[0])
                IndexRangeVisitor<N - 1>::template m_run<1>(f, begin, end, i);
        }
    }

private:
    // Run the D^th nested loop (where `D = 0` would be outermost),
    // which consists of `N` nested loops.
    // Note: these inner loops are never parallelized.
    template<size_t D, class F, class IndexContainer>
    static void m_run(const F &f, const IndexContainer &begin, const IndexContainer &end, IndexContainer &i) {
        for (i[D] = begin[D]; i[D] < end[D]; ++i[D])
            IndexRangeVisitor<N - 1>::template m_run<D + 1>(f, begin, end, i);
    }

    template<size_t N_, bool Parallel_>
    friend struct IndexRangeVisitor;

    template<size_t N_, typename TLD>
    friend struct IndexRangeVisitorThreadLocal;
};

// Base case (body of innermost loop): apply visitor!
template<>
struct IndexRangeVisitor<0, false> {
private:
    template<size_t D, class F, class IndexContainer>
    static void m_run(const F &f, const IndexContainer &/* begin */, const IndexContainer &/* end */, IndexContainer &i) {
        static_assert(D == IndexContainer::RowsAtCompileTime, "Index count mismatch");
        f(i);
    }
    template<size_t N_, bool Parallel_>
    friend struct IndexRangeVisitor;
};

// Visit the corners of the ND index hypercube [0, 1]^D
template<size_t N>
struct HypercubeCornerVisitor;

template<>
struct HypercubeCornerVisitor<1> {
    template<class F>
    static void run(const F &f) {
        using IndexContainer = std::decay_t<typename function_traits<F>::template arg<0>::type>;
        IndexContainer i;
        for (i[0] = 0; i[0] < 2; ++i[0]) f(i);
    }
};

template<>
struct HypercubeCornerVisitor<2> {
    template<class F>
    static void run(const F &f) {
        using IndexContainer = std::decay_t<typename function_traits<F>::template arg<0>::type>;
        IndexContainer i;
        for (    i[0] = 0; i[0] < 2; ++i[0])
            for (i[1] = 0; i[1] < 2; ++i[1])
                f(i);
    }
};

template<>
struct HypercubeCornerVisitor<3> {
    template<class F>
    static void run(const F &f) {
        using IndexContainer = std::decay_t<typename function_traits<F>::template arg<0>::type>;
        IndexContainer i;
        for (        i[0] = 0; i[0] < 2; ++i[0])
            for (    i[1] = 0; i[1] < 2; ++i[1])
                for (i[2] = 0; i[2] < 2; ++i[2])
                    f(i);
    }
};

#endif /* end of include guard: INDEXUTILS_HH */
