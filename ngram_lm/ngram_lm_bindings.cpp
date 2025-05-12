// Compilation string: c++ -O3 -Wall -shared -std=c++17 -fPIC $(uv run python3 -m pybind11 --includes) ngram_lm_bindings.cpp -o ngram_lm.cpython-39-x86_64-linux-gnu.so
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For automatic conversion of STL containers like std::vector and std::string
#include "ngram_lm.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ngram_lm, m) // The module name will be 'ngram_lm' in Python
{
    m.doc() = "Python bindings for the NGramLM C++ class"; // Optional module docstring

    py::class_<NGramLM>(m, "NGramLM")
        .def(py::init<int>(), py::arg("n"),
             "Constructor for NGramLM. Takes the n-gram order 'n' as an argument.")

        .def("fit", &NGramLM::fit, py::arg("sequences"),
             "Fits the n-gram model to the given sequences. "
             "Expects a list of lists of strings (e.g., [['a', 'b'], ['b', 'c']]).")

        .def("get_ngram_probability", &NGramLM::getNgramProbability,
             py::arg("ngram"),
             py::arg("method") = "kneser-ney",
             py::arg("discount") = 0.75,
             "Get the probability of a single n-gram. "
             "Expects a list of strings for the ngram. "
             "Optional 'method' (default 'kneser-ney') and 'discount' (default 0.75).")

        .def("get_ngram_probabilities", &NGramLM::getNgramProbabilities,
             py::arg("ngrams_batch"),
             py::arg("method") = "kneser-ney",
             py::arg("discount") = 0.75,
             "Get probabilities for a batch of n-grams concurrently. "
             "Expects a list of lists of strings for the batch. "
             "Optional 'method' (default 'kneser-ney') and 'discount' (default 0.75).");
}