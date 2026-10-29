Development
===========

Contributions are welcome. mf6rtm uses `pixi <https://pixi.sh>`_ for fast,
reproducible development environments.

Setup
-----

.. code-block:: bash

   # Fork, then clone your fork
   git clone https://github.com/YOUR-USERNAME/mf6rtm.git
   cd mf6rtm

   # Install the dev environment with all dependencies
   pixi install

Common tasks
------------

.. code-block:: bash

   pixi run test          # run the test suite
   pixi run test-cov      # run tests with coverage
   pixi run lint          # run linting
   pixi run -e py311 test  # test against a specific Python version

Building the docs
-----------------

.. code-block:: bash

   pixi run -e docs build-docs

Contributing
------------

1. Open an `issue <https://github.com/p-ortega/mf6rtm/issues>`_ to discuss
   bugs or feature ideas.
2. Branch off ``develop``, keep changes focused, and add tests where it makes
   sense.
3. Make sure ``pixi run test`` and ``pixi run lint`` pass, then open a pull
   request.
