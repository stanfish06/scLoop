#ifndef RIPSER_H
#define RIPSER_H

#include <vector>
#include <queue>
#include <cmath>
#include <unordered_map>

typedef float value_t;
typedef int64_t index_t;
typedef int16_t coefficient_t;

template <class Key, class T, class H, class E>
class hash_map_imp;
template <class Key, class T, class H, class E>
class hash_map : public hash_map_imp<Key, T, H, E> {};

typedef index_t entry_t;
typedef std::pair<value_t, index_t> diameter_index_t;
struct diameter_entry_t : std::pair<value_t, entry_t> {
    using std::pair<value_t, entry_t>::pair;
};

template <typename Entry>
struct greater_diameter_or_smaller_index {
    bool operator()(const Entry& a, const Entry& b);
};
enum compressed_matrix_layout { LOWER_TRIANGULAR, UPPER_TRIANGULAR };

template <compressed_matrix_layout Layout>
class compressed_distance_matrix {
public:
    std::vector<value_t> distances;
    std::vector<value_t*> rows;

    compressed_distance_matrix(std::vector<value_t>&& _distances);

    template <typename DistanceMatrix>
    compressed_distance_matrix(const DistanceMatrix& mat);

    value_t operator()(const index_t i, const index_t j) const;
    size_t size() const { return rows.size(); }
    void init_rows();
};

typedef compressed_distance_matrix<LOWER_TRIANGULAR> compressed_lower_distance_matrix;
template <typename ValueType>
class compressed_sparse_matrix;

/*
 These are the default results returned from ripser
*/
typedef struct {
    std::vector<std::vector<value_t>> births_and_deaths_by_dim;
    std::vector<std::vector<std::vector<int>>> cocycles_by_dim;
    int num_edges;
} ripserResults;

/*
 Data structure for returning dimension 2 boundary matrix
*/
typedef struct {
    std::vector<std::vector<index_t>> triangle_vertices;
    std::vector<value_t> triangle_diameters;
} boundaryMatrixResults;

/*
  Main data structure
*/
template <typename DistanceMatrix>
class ripser {
    const DistanceMatrix dist;
    index_t n, dim_max;
    const value_t threshold;
    const float ratio;
    const coefficient_t modulus;
    mutable std::vector<diameter_entry_t> cofacet_entries;
    const int do_cocycles;
public:
    mutable std::vector<std::vector<value_t>> births_and_deaths_by_dim;
    mutable std::vector<std::vector<std::vector<int>>> cocycles_by_dim;

    struct entry_hash {
        std::size_t operator()(const entry_t& e) const;
    };

    struct equal_index {
        bool operator()(const entry_t& e, const entry_t& f) const;
    };

    typedef hash_map<entry_t, size_t, entry_hash, equal_index> entry_hash_map;

    ripser(DistanceMatrix&& _dist, index_t _dim_max, value_t _threshold,
           float _ratio, coefficient_t _modulus, int _do_cocycles);

    index_t get_edge_index(const index_t i, const index_t j) const;

    template <typename OutputIterator>
    OutputIterator get_simplex_vertices(index_t idx, const index_t dim,
                                        index_t n, OutputIterator out) const;

    class simplex_coboundary_enumerator;

    void assemble_columns_to_reduce(std::vector<diameter_index_t>& simplices,
                                    std::vector<diameter_index_t>& columns_to_reduce,
                                    entry_hash_map& pivot_column_index, index_t dim);

    value_t get_vertex_birth(index_t i);

    void compute_dim_0_pairs(std::vector<diameter_index_t>& edges,
                             std::vector<diameter_index_t>& columns_to_reduce);

    template <typename Column>
    diameter_entry_t init_coboundary_and_get_pivot(
        const diameter_entry_t simplex, Column& working_coboundary,
        const index_t& dim, entry_hash_map& pivot_column_index);

    template <typename Column>
    void add_simplex_coboundary(const diameter_entry_t simplex,
                                const index_t& dim,
                                Column& working_reduction_column,
                                Column& working_coboundary);

    template <typename Column>
    void add_coboundary(compressed_sparse_matrix<diameter_entry_t>& reduction_matrix,
                        const std::vector<diameter_index_t>& columns_to_reduce,
                        const size_t index_column_to_add, const coefficient_t factor,
                        const size_t& dim, Column& working_reduction_column,
                        Column& working_coboundary);

    typedef std::priority_queue<
        diameter_entry_t, std::vector<diameter_entry_t>,
        greater_diameter_or_smaller_index<diameter_entry_t>> working_t;

    mutable diameter_entry_t cocycle_e;
    mutable std::vector<index_t> cocycle_simplex;
    mutable std::vector<int> thiscocycle;

    void compute_cocycles(working_t cocycle, index_t dim);

    void compute_pairs(std::vector<diameter_index_t>& columns_to_reduce,
                       entry_hash_map& pivot_column_index, index_t dim);

    std::vector<diameter_index_t> get_edges();

    void compute_barcodes();

    void copy_results(ripserResults& res);

    std::vector<std::pair<std::vector<index_t>, value_t>>
    assemble_full_dim_2_boundary_matrix();
};

// main api
ripserResults rips_dm(float* D, int N, int modulus, int dim_max,
                      float threshold, int do_cocycles);
/**
 * @brief compute PH with sparse distance matrix
 * @details diagonal should probably be 0, but it appears that vertex can have nonzero birth as well
 * @param I row idx
 * @param J col idx
 * @param V value
 * @param NEdges number of nonzero edges
 * @param N number of vertices
 * @param modulus 2 if F2
 * @param dim_max 1 if want only H0 and H1
 * @param threshold edge threshold
 * @param do_cycles 1 if want to return cocycles
 */
ripserResults rips_dm_sparse(int* I, int* J, float* V, int NEdges, int N,
                             int modulus, int dim_max, float threshold,
                             int do_cocycles);

/**
 * @brief Compute dimension 2 boundary matrix without redundant columns
 * @details For each tetrahedron, only 3 of the 4 triangular faces are included,
 *          as the 4th can be derived from the other 3 (in F2 homology).
 * @param I row indices of sparse distance matrix
 * @param J column indices of sparse distance matrix
 * @param V values of sparse distance matrix
 * @param NEdges number of nonzero entries in distance matrix
 * @param N number of vertices
 * @param threshold maximum diameter for simplices to include
 * @return boundaryMatrixResults containing triangle vertices and diameters
 */
boundaryMatrixResults get_boundary_matrix_sparse(int* I, int* J, float* V,
                                                  int NEdges, int N,
                                                  float threshold);

// TODO: need function to extract cocycles
#endif
