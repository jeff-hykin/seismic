#!/usr/bin/env sh
echo --% >/dev/null;: ' | out-null
<#'

#
# for not-Windows operating systems
#
deno run -A https://esm.sh/gh/jeff-hykin/bite@e070c4af1e/vite/bin/vite.js build
deno run -A https://esm.sh/gh/jeff-hykin/html-bundle@0.0.3.0/main/html-bundle.js --inplace ./dist/index.html

exit #>

#
# for windows (powershell)
#
deno run -A https://esm.sh/gh/jeff-hykin/bite@e070c4af1e/vite/bin/vite.js build
deno run -A https://esm.sh/gh/jeff-hykin/html-bundle@0.0.3.0/main/html-bundle.js --inplace ./dist/index.html
# deno run -A https://deno.land/x/html-bundle/main/html-bundle.js --inplace ./docs/index.html 