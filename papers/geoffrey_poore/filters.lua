function Str(elem)
    local startIndex, endIndex, maybeMatch = string.find(elem.text, '%[(%w+)%]')
    if maybeMatch then
        local newInlines = pandoc.Inlines{}
        if startIndex > 1 then
            newInlines:insert(pandoc.Str(string.sub(elem.text, 1, startIndex-1)))
        end
        newInlines:insert(pandoc.RawInline('rst', ':cite:`' .. maybeMatch .. '`'))
        if endIndex < elem.text:len() then
            newInlines:insert(pandoc.Str(string.sub(elem.text, endIndex+1)))
        end
        return newInlines
    end
end
