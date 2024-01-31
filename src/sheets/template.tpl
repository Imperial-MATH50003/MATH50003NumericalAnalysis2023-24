\documentclass[12pt,a4paper]{article}

\usepackage[a4paper,text={16.5cm,25.2cm},centering]{geometry}
\usepackage{lmodern}
\usepackage{amssymb,amsmath}
\usepackage{bm}
\usepackage{graphicx}
\usepackage{microtype}
\usepackage{hyperref}
{{#:tex_deps}}
{{{ :tex_deps }}}
{{/:tex_deps}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{1.2ex}




\hypersetup
       {   pdfauthor = { {{{:author}}} },
           pdftitle={ {{{:title}}} },
           colorlinks=TRUE,
           linkcolor=black,
           citecolor=blue,
           urlcolor=blue
       }

{{#:title}}
\title{ {{{ :title }}} }
{{/:title}}

{{#:author}}
\author{ {{{ :author }}} }
{{/:author}}

{{#:date}}
\date{ {{{ :date }}} }
{{/:date}}

{{ :highlight }}

\def\endash{–}
\def\bbD{ {\mathbb D} }
\def\bbZ{ {\mathbb Z} }

\def\x{ {\vc x} }
\def\a{ {\vc a} }
\def\b{ {\vc b} }
\def\e{ {\vc e} }
\def\f{ {\vc f} }
\def\u{ {\vc u} }

\input{somacros}

\begin{document}

{{#:title}}\maketitle{{/:title}}

{{{ :body }}}

\end{document}