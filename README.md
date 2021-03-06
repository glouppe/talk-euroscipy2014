Accelerating Random Forests in Scikit-Learn
===========================================

EuroScipy 2014 (http://euroscipy.org/2014)

_Permanent URL (PDF):_  http://hdl.handle.net/2268/171887

_Contact:_ Gilles Louppe (@glouppe, <g.louppe@gmail.com>)

---

Random Forests are without contest one of the most robust, accurate and versatile tools for solving machine learning tasks. Implementing this algorithm properly and efficiently remains however a challenging task involving issues that are easily overlooked if not considered with care. In this talk, we present the Random Forests implementation developed within the Scikit-Learn machine learning library. In particular, we describe the iterative team efforts that led us to gradually improve our codebase and eventually make Scikit-Learn's Random Forests one of the most efficient implementations in the scientific ecosystem, across all libraries and programming languages. Algorithmic and technical optimizations that have made this possible include:

- An efficient formulation of the decision tree algorithm, tailored for Random Forests;
- Cythonization of the tree induction algorithm;
- CPU cache optimizations, through low-level organization of data into contiguous memory blocks;
- Efficient multi-threading through GIL-free routines;
- A dedicated sorting procedure, taking into account the properties of data;
- Shared pre-computations whenever critical.

Overall, we believe that lessons learned from this case study extend to a broad range of scientific applications and may be of interest to anybody doing data analysis in Python.
