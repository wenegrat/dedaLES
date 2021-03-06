.. math::
    \newcommand{\b}[1]{\boldsymbol{#1}}
    \newcommand{\r}[1]{\mathrm{#1}}
    \newcommand{\bz}{\b{z}}
    \newcommand{\bu}{\b{u}}
    \newcommand{\bcdot}{\b{\cdot}}
    \newcommand{\d}{\partial}

    \newcommand{\p}{\, .}
    \newcommand{\c}{\, ,}

.. _constant Smagorinsky:

Constant Smagorinsky
====================

In the first-order 'constant Smagorinsky' turbulence closure, the subgrid stress
:math:`F^\bu_{ij}` defined in terms of the resolved rate of strain tensor
:math:`S_{ij} = \tfrac{1}{2} \left ( \d_i u_j + \d_j u_i \right )` 
and an eddy viscosity :math:`\nu_e`:

.. math::

    F^\bu_{ij} = 2 \nu_e S_{ij} \p

In dedaLES, the eddy viscosity :math:`\nu_e` is defined via a slight
generalization of traditional constant Smagorinsky,

.. math::

    \nu_e = \left [ \Delta_{\r{const}}^2 + \left ( C \Delta \right )^2 \right ] | \b{S} | \, ,

where :math:`\Delta_{\r{const}}` is a constant 'filter width', 
:math:`C` is the Poincaré constant,
and :math:`\Delta` is a filter width defined by
some multiple of the grid resolution, and thus dependent on position 
within the chosen grid in general.
The invariant of the resolved strain tensor :math:`|\b{S}|` is

.. math::

    | \b{S} | \equiv \sqrt{ 2 S_{ij} S_{ji} } \, .

Note that :math:`S_{ij}` is symmetric, so that :math:`S_{ij} = S_{ji}`.
The subgrid buoyancy flux is

.. math::

    \b{F}^b = -\kappa_e \nabla b \c

with :math:`\kappa_e = \nu_e / Pr_e` for effective turbulent Prandtl number 
:math:`Pr_e`.

References
----------

- `wikipedia`_
- `Smagorinsky 1963`_
- `Vreugdenhil and Taylor 2018`_ 

.. _wikipedia: https://en.wikipedia.org/wiki/Large_eddy_simulation#Smagorinsky%E2%80%93Lilly_model
.. _Smagorinsky 1963: https://journals.ametsoc.org/doi/abs/10.1175/1520-0493%281963%29091%3C0099%3AGCEWTP%3E2.3.CO%3B2
.. _Vreugdenhil and Taylor 2018: https://aip.scitation.org/doi/abs/10.1063/1.5037039
