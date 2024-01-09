#####
# extras
#####

using Weave

nkwds = (out_path="notes/", jupyter_path="$(homedir())/.julia/conda/3/x86_64/bin/jupyter", nbconvert_options="--allow-errors")
notebook("src/notes/A.Julia.jmd"; nkwds...)

#####
# notes
#####

using Weave


function replacetheorem(path, thrm, Thrm)
    str = read(path, String)
    str = replace(str, Regex("\\\\textbf{$Thrm \\(([^}]*)\\)}(.*?)\\\\textbf{Proof}", "s") => SubstitutionString("\\\\begin{$thrm}[\\1]\\2\\\\end{$thrm}\\n\\\\textbf{Proof}"))
    write(path, str)
end

function replacedefinition(path, thrm, Thrm)
    str = read(path, String)
    str = replace(str, Regex("\\\\textbf{$Thrm \\(([^}]*)\\)}(.*?)\\\\ensuremath{\\\\QED}", "s") => SubstitutionString("\\\\begin{$thrm}[\\1]\\2\\\\end{$thrm}"))
    write(path, str)
end

function compilenotes(filename)
    weave("src/notes/$filename.jmd"; out_path="notes/", doctype="md2tex", template="src/notes/template.tpl")
    path = "notes/$filename.tex"
    replacetheorem(path, "theorem", "Theorem")
    replacetheorem(path, "lemma", "Lemma")
    replacetheorem(path, "proposition", "Proposition")
    replacedefinition(path, "example", "Example")
    replacedefinition(path, "definition", "Definition")
    # work around double newline before equation
    write(path, replace(read(path, String), "\n\n\\[" => "\n\\["))
    # work around meeq 
    write(path, replace(read(path, String), r"\\\[\n\\meeq\{(.*?)\}\n\\\]"s => s"\\meeq{\1}"))
end

compilenotes("I.1.RectangularRule")
compilenotes("I.2.DividedDifferences")
compilenotes("I.3.DualNumbers")


###
# sheets
###

function compilesheet(filename)
    write("sheets/$filename.jmd", replace(read("src/sheets/$(filename)s.jmd", String), r"\*\*SOLUTION\*\*(.*?)\*\*END\*\*"s => ""))
    weave("sheets/$filename.jmd"; out_path="sheets/", doctype="md2tex", template="src/sheets/template.tpl")
    path = "sheets/$filename.tex"
    # work around double newline before equation
    write(path, replace(read(path, String), "\n\n\\[" => "\n\\["))
    # work around meeq 
    write(path, replace(read(path, String), r"\\\[\n\\meeq\{(.*?)\}\n\\\]"s => s"\\meeq{\1}"))
end

for k = 1:2
    compilesheet("sheet$k")
end

# notebook("sheets/sheet1.jmd"; pkwds...)
# notebook("src/sheets/sheet1s.jmd"; pkwds...)


#####
# labs
#####

import Literate

# Make Labs
for k = 1:2
    write("labs/lab$k.jl", replace(replace(read("src/labs/lab$(k)s.jl", String), r"## SOLUTION(.*?)## END"s => "")))
    Literate.notebook("labs/lab$k.jl", "labs/"; execute=false)
end


# Make Solutions
# Literate.notebook("src/labs/lab1s.jl", "labs/"; execute=false)

