#!/bin/bash
# Tracked-changes PDF: previous version (default ../manuscript_v2; pass another folder as $1) vs current.
# Added text blue, deleted text red strikethrough.
set -e
cd "$(dirname "$0")"
BASE=${1:-../manuscript_v2}
mkdir -p diff
for f in sec_abstract sec_intro_methods sec_results; do
  latexdiff --type=CFONT --exclude-textcmd="section,subsection,caption" \
    $BASE/$f.tex $f.tex > diff/$f.tex 2>/dev/null
done
cp access.tex diff/access_diff.tex
sed -i 's/\\input{sec_\(abstract\|intro_methods\|results\).tex}/\\input{diff\/sec_\1.tex}/' diff/access_diff.tex
# colours per the authors' convention: additions blue, deletions red with strikethrough
cat > diff/diffpreamble.tex <<'EOT'
\usepackage{ulem}\normalem
\providecommand{\DIFadd}[1]{{\color{blue}#1}}
\providecommand{\DIFdel}[1]{{\color{red}\sout{#1}}}
EOT
sed -i 's/\\input{results_macros.tex}/\\input{results_macros.tex}\n\\input{diff\/diffpreamble.tex}/' diff/access_diff.tex
for f in diff/sec_*.tex; do
  python3 - "$f" <<'PY'
import re,sys
p=sys.argv[1]; s=open(p).read()
# drop latexdiff's own preamble (we supply our own) and normalise macros
s=re.sub(r"%DIF PREAMBLE EXTENSION.*?%DIF END PREAMBLE EXTENSION ADDED BY LATEXDIFF\n", "", s, flags=re.S)
s=s.replace("\\DIFaddbegin","").replace("\\DIFaddend","").replace("\\DIFdelbegin","").replace("\\DIFdelend","")
s=s.replace("\\DIFaddFL","\\DIFadd").replace("\\DIFdelFL","\\DIFdel")
open(p,"w").write(s)
PY
done
pdflatex -interaction=nonstopmode -output-directory=diff diff/access_diff.tex >/dev/null || true
pdflatex -interaction=nonstopmode -output-directory=diff diff/access_diff.tex >/dev/null || true
grep -c "^!" diff/access_diff.log || true
ls -la diff/access_diff.pdf
