## Development setup
1. **Install the dependencies**:
   ```bash
    pip install mystmd
    snap install typst
   ```

2. Two options for **showing the document**:
    - In-browser document:
      ```bash
      myst start
      ```
      and open http://localhost:3000 in your browser.

    - PDF document:
      ```bash
      myst build --pdf
      ```