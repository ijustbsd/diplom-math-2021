THEME = Математическая модель импульсного погружателя с учетом малой асимметрии
STUDENT = С.Д. Бабошин
DEGREE = д. ф.-м. н., проф.
DIRECTOR = М.И. Каменский

SED = "s/{{theme}}/${THEME}/; s/{{student}}/${STUDENT}/; s/{{degree}}/${DEGREE}/; s/{{director}}/${DIRECTOR}/"

all:
	# титульный лист
	sed -e ${SED} titlepage.fodt > tp-output.fodt
	libreoffice --headless --convert-to pdf tp-output.fodt

	# .tex
	pdflatex diplom.tex
	biber diplom
	pdflatex diplom.tex
	pdflatex diplom.tex
	mv diplom.pdf Бабошин.pdf
	evince Бабошин.pdf &

presentation:
	pdflatex presentation.tex
	pdflatex presentation.tex
	evince presentation.pdf &

clean:
	rm -f *.aux *.log *.out *.toc *.pdf *.bbl *.bcf *.blg *.xml *.nav *.snm
	rm -f images/*.pdf
	rm -f tp-output.*
