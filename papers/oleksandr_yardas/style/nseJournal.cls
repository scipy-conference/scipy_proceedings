\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{nseJournal}[2018/06/25 Nuclear Science and Engineering Journal]

\newif\iffootnoteAtEnd \footnoteAtEndfalse
\DeclareOption{footnoteAtEnd}{\footnoteAtEndtrue}
\DeclareOption*{\PassOptionsToPackage{\CurrentOption}{article}}
\ProcessOptions \relax

\LoadClass{article}

\RequirePackage[margin=1.3in]{geometry}                                         %set size of all margins
\RequirePackage{setspace}                                                       %option 2 for making text double-spaced
\RequirePackage{indentfirst}                                                    %makes first paragraph of section indented (non-first are by default)
\RequirePackage[explicit]{titlesec}                                             %package needed for next lines
\RequirePackage{lastpage}                                                       %package which returns number of last page (same as number of pages)
\RequirePackage[figure,table]{totalcount}                                       %package which counts the number of tables and/or figures
\RequirePackage{amsmath}                                                        %enable `align' equation types
\RequirePackage{graphicx}                                                       %enable figures
\RequirePackage{multirow}                                                       %enable `multirow' capability in tables

\RequirePackage{caption}                                                        %enables subfigures
\RequirePackage[labelformat=simple]{subcaption}                                 %enables subfigure captions 
    \captionsetup[table]{labelsep=newline,name=TABLE}                           %sets table caption formatting options to meet NSE requirements
    \captionsetup[figure]{name=Fig.,labelsep=period}                            %sets figure caption options to meet NSE requirements
\RequirePackage[colorlinks=true, allcolors=blue, pdfencoding=auto]{hyperref}    %reference and citation hyperlinking linking
\RequirePackage[]{footmisc}                                                     %enables double spaced footnotes
    \renewcommand{\footnotelayout}{\doublespacing}                              %double spacing of footnotes
\RequirePackage{xstring}
\RequirePackage{etoolbox}
\RequirePackage{endnotes}                                                       %enables endnote page



\renewcommand\thesection{\Roman{section}}                                   %set capital Roman numeral section headings
\renewcommand\thesubsection{\thesection.\Alph{subsection}}                  %set capital Aramaic letters subsection headings
\renewcommand\thesubsubsection{\thesubsection.\arabic{subsubsection}}       %set capital Arabic numbers subsubsection headings

\renewcommand*\thetable{\Roman{table}}                                      %set capital Roman numeral table numeration

\titleformat{\section}{\bfseries}{\thesection.}{1em}{\MakeUppercase{#1}}    %makes section headings bold and upper case characters
\titleformat{\subsection}{\bfseries}{\thesubsection.}{1em}{#1}              %makes subsection headings bold
\titleformat{\subsubsection}{\itshape}{\thesubsubsection.}{1em}{#1}         %makes subsubsection headings in italics

\renewcommand{\thefootnote}{\alph{footnote}}                                %sets labeling of footnotes

\renewcommand*\thesubfigure{(\alph{subfigure})}                             %enables proper labeling of subfigures

\DeclareMathOperator{\diff}{d}                                              %can define operators to use in math mode, example in Eq. 2
\DeclareMathOperator{\erf}{erf}

\doublespacing                                                              %set double space
\setlength{\parindent}{25pt}                                                %set size of indentation (15pt is default)

\iffootnoteAtEnd
    \let\footnote=\endnote                                                  %sets footnotes to endnotes
    \renewcommand*{\theendnote}{\alph{endnote}}                             %labels endnotes alphabetically
    \renewcommand{\notesname}{Footnotes}                                    %renames endnotes title to footnotes
\fi
\newcommand\notes{
    \iffootnoteAtEnd
        \pagebreak
        \theendnote
        Hey some notes!
    \fi
}
\let\Oldenddocument\enddocument%
\def\enddocument{\notes\Oldenddocument}%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Title Page Macros
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def\@correspondingEmail{\@empty}
\newcommand\correspondingAuthor[1]{#1$^{\ast,}$}
\newcommand\correspondingEmail[1]{\def\@correspondingEmail{#1}}
\newcommand\showcorrespondingEmail{$^\ast$Email: \href{mailto:\@correspondingEmail}{\@correspondingEmail}}

\def\@authors{\@empty}
\newcommand\addAuthor[2]{
    \StrSubstitute{\@authors}{and}{}[\@authors]
    \StrSubstitute{\@authors}{$^}{,$^}[\@authors]
    \StrSubstitute{\@authors}{$,$}{$$}[\@authors]
    \StrSubstitute{\@authors}{,,}{,}[\@authors]
    \edef\@authors{\@authors\if\@authors\@empty\else\ and \fi #1$^{\text{#2}}$}
    % \noindent
    % \@authors\\
}
\newcommand\showAuthors{\@authors}

\def\@affiliations{\@empty}
\newcommand\addAffiliation[2]{
    \let\\\tabularnewline
    \edef\@affiliations{\@affiliations\tabularnewline\tabularnewline$^{#1}$#2}
}
\newcommand\showAffiliations{
    \begin{itshape}
    \renewcommand{\arraystretch}{0.5}
    \begin{tabular}[t]{c}
        \@affiliations
    \end{tabular}
    \end{itshape}
}

\newcommand\titlePage{
    \author{
        \vspace{20mm}
        \\
        \begin{minipage}[c]{\textwidth}
            \centering
            \showAuthors
        \end{minipage}
        \\[4pt]
        \showAffiliations\\
        \showcorrespondingEmail
    }
    \date{
        \vfill
        \renewcommand{\arraystretch}{0.8}
        \begin{tabular}[t]{rl}
        Number of pages:& \pageref*{LastPage} \\
        Number of tables:& \totaltables \\
        Number of figures:& \totalfigures
        \end{tabular}
    }
    \clearpage\maketitle
    \thispagestyle{empty}
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Format Abstract environment
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\def\@keywords{\@empty}
\newcommand\addKeyword[1]{
    \edef\@keywords{\@keywords\if\@keywords\@empty\else,\ \fi#1}
}
\newcommand\showKeywords{\par\vspace{1em}\noindent\textbf{Keywords} --- \@keywords}
\renewenvironment{abstract}{\pagebreak~\vfill\begin{center}\textbf{Abstract}\end{center}}{\showKeywords\vfill\pagebreak}
