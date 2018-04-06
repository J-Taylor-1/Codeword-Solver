# Codeword-Solver
Solves images of Codeword Puzzles, by reading the image's characters and inferring the words from a dictionary based on the SUBTLEX-UK corpus.

Run the program on an example image or custom image with:

python cw_solve.py filepath.png xCells yCells cellProp borderWidth
Where:
  cellProp refers to the proportion of the cell that is taken up by the number at the top of each cell
  borderWidth refers to the width of the border between cells, which should be adjusted for (usually ~0.1)
  
 For example, to analyse example image "cw02.png", try:
 
 "python cw_solve.py cw02.png 13 13 0.35 0.1"
 
 
 Note that the script uses pytesseract, which relies on PIL/Pillow and Google Tesseract OCR.
