if not FORMAT:match 'latex' then
    return {}
  end

  function Table (tbl)
    local opts = PANDOC_WRITER_OPTIONS
    opts.template = nil
    local tex = pandoc.write(
      pandoc.Pandoc{tbl},
      'latex'
    )
    local caption = tex:match '.*(\\caption%b{}).*' or ''
    tex = tex
      :gsub('\\toprule%(%)\n.*\\endfirsthead\n', '')
      :gsub('@%{%}', '')
      :gsub(
        '^\\begin%{longtable%}%[%]',
        '\\begin{table}' ..
        caption ..
        '\\begin{center}\\begin{tabularx}{\\tablewidth}'
      )
      :gsub(
        '\n\\end%{longtable%}',
        '\\end{tabularx}\\end{center}\\end{table}\\vspace{2mm}'
      )
      :gsub('\n\\toprule%(%)\n', '\n\\toprule\n')
      :gsub('\n\\caption%b{}', '\n')
      :gsub('\n\\endhead', '\n')
      :gsub('\n\\midrule%(%)\n', '\n\\midrule\n')
      :gsub('\n\\bottomrule%(%)', '\n\\bottomrule')
      :gsub('\n\\tabularnewline\n', '\n')
    return pandoc.RawBlock('latex', tex)
  end
