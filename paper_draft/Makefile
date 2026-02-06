# Makefile for LaTeX paper compilation

MAIN = main
LATEX = pdflatex
BIBTEX = bibtex
LATEX_FLAGS = -interaction=nonstopmode -halt-on-error

.PHONY: all clean view

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib sections/*.tex commands/*.tex
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	$(BIBTEX) $(MAIN) || true
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
	rm -f sections/*.aux
	rm -f $(MAIN).pdf

view: $(MAIN).pdf
	open $(MAIN).pdf || xdg-open $(MAIN).pdf || start $(MAIN).pdf

# Quick build (single pass, no bibliography)
quick:
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
