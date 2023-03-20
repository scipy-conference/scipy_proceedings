local ntables = 0
local nfigures = 0
local nequations = 0

local table_labels = {}
local figure_labels = {}
local equation_labels = {}
local labels = {}

local stringify = pandoc.utils.stringify
local get_tag = function (x) return x.t end
local List = pandoc.List

local function collect_table_labels (tbl)
  ntables = ntables + 1
  return tbl:walk {
    Span = function (span)
      local label = span.attributes.label
      if label then
        table_labels[label] = tostring(ntables)
        labels[label] = {pandoc.Str(tostring(ntables))}
        return pandoc.RawInline('latex', '\\label{' .. label .. '}')
      end
    end
  }
end

local function collect_figure_labels (para)
  -- ensure that we are looking at an implicit figure
  if #para.content ~= 1 or para.content[1].t ~= 'Image' then
    return nil
  end
  local image = para.content[1]
  nfigures = nfigures + 1
  return pandoc.Para{
    image:walk {
      Span = function (span)
        local label = span.attributes.label
        if label then
          figure_labels[label] = tostring(nfigures)
          labels[label] = {pandoc.Str(tostring(nfigures))}
          return pandoc.RawInline('latex', '\\label{' .. label .. '}')
        end
      end
    }
  }
end

local function collect_equation_tags (span)
  local label = span.attributes.label
  if span.classes[1] == 'equation' and label then
    nequations = nequations + 1
    equation_labels[label] = tostring(nequations)
    labels[label] = {pandoc.Str(tostring(nequations))}
  end
end

local function normalize_label_span (span)
  if span.classes == List{'label'} and span.identifier == '' then
    span.identifier = span.attributes.label or stringify(span)
    span.attributes.label = span.identifier
    span.classes = {}
    span.content = {}
  end
  if span.attributes.label and span.identifier == '' then
    span.identifier = span.attributes.label
  end
  if #span.content == 1 then
    local formula = span.content[1]
    if formula.t == 'Math' and formula.mathtype == 'DisplayMath' then
      span.classes:insert('equation')
    end
  end
  return span
end

local function resolve_ref_number (span)
  if span.classes == List{'ref'} then
    local target = pandoc.utils.stringify(span)
    if FORMAT:match 'latex' then
      return pandoc.RawInline('latex', '\\ref{' .. target .. '}')
    else
      return pandoc.Link(labels[target] or '', '#' .. target)
    end
  end
end

-- local function resolve_citation (str)
--   if FORMAT:match 'latex' then
--     if string.find(':cite:`') then
--         local text = str :gsub:('`', '') :gsub:(':cite:', '')
--         local text = string.format('\\cite{%s}', no_cite)
--         return pandoc.RawBlock('latex', text)
--     end
--   end
-- end

local function latex_equation (span)
  if span.classes == List{'equation'} then
    local formula = #span.content == 1 and span.content[1] or nil
    if formula and formula.t == 'Math' then
      local env = span.type or 'equation'
      local label = span.attributes.label
        and string.format('\\label{%s}', span.attributes.label)
        or ''
      return pandoc.RawInline(
        'latex',
        string.format(
          [[\begin{%s}%s%s\end{%s}]], env, formula.text, label, env
        )
      )
    end
  end
end

if not FORMAT:match 'latex' then
  latex_equation = nil
end

return {
  { Span = normalize_label_span },
  {
    Table = collect_table_labels,
    Para = collect_figure_labels,
    Span = collect_equation_tags,
  },
  { Span = resolve_ref_number },
--   { Str = resolve_citation },
  { Span = latex_equation },
}
