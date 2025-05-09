// Compilation string: c++ -O3 -Wall -shared -std=c++17 -fPIC $(uv run python3 -m pybind11 --includes) /usr/lib/libgtest.a /usr/lib/libgtest_main.a trie_bindings.cpp -o trie.cpython-39-x86_64-linux-gnu.so
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "trie.hpp"

namespace py = pybind11;

PYBIND11_MODULE(trie, m)
{
    py::class_<Trie>(m, "Trie")
        .def(py::init<>()) // Bind the constructor
        .def("add", &Trie::add, "Add a sequence to the trie")
        .def("getCounts", [](Trie &self, const std::vector<std::string> &query)
             { return self.getCounts(query); }, "Get counts for a query sequence")
        .def("getCountsBatch", [](Trie &self, const std::vector<std::vector<std::string>> &queries)
             { return self.getCountsBatch(queries); }, "Get counts for multiple query sequences");
}