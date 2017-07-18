function proc_versions() {
   var versions = range(2017, 2008)
   var proc_url = 'http://conference.scipy.org/proceedings/scipy';
   document.write('<ul id="navibar">');

   for (i=0; i < versions.length; i++) {
       document.write('<li class="wikilink">');
       document.write('<a href="' + proc_url + versions[i] + '">SciPy ' + versions[i] +  '</a>');
       document.write('</li>');
   }

   document.write('</ul>');
}
var range = function(start, end, step) {
    var range = [];
    var typeofStart = typeof start;
    var typeofEnd = typeof end;

    if (step === 0) {
        throw TypeError("Step cannot be zero.");
    }

    if (typeofStart == "undefined" || typeofEnd == "undefined") {
        throw TypeError("Must pass start and end arguments.");
    } else if (typeofStart != typeofEnd) {
        throw TypeError("Start and end arguments must be of same type.");
    }

    typeof step == "undefined" && (step = 1);

    if (end < start) {
        step = -step;
    }

    if (typeofStart == "number") {

        while (step > 0 ? end >= start : end <= start) {
            range.push(start);
            start += step;
        }

    } else {
        throw TypeError("Only number types are supported");
    }

    return range;

}
