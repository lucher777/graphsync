# Contributing to graphsync

Contributions, bug reports, and mathematical improvements are welcome.

## Development Workflow

1. Fork the repository and create a topic branch.
2. Ensure all code adheres to PEP 8 and includes complete Google-style docstrings for all public classes and functions.
3. Run the test suite:
   ```bash
   pytest
   ```
4. Verify strict numerical parity (`graphsync.verify_parseval_and_derivatives` and `graphsync.verify_crlb_fim_parity`) before opening a pull request.
