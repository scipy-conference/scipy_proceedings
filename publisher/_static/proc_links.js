function proc_versions() {
   var versions = range('2017', '2008')
   var proc_url = 'http://conference.scipy.org/proceedings/scipy';
   document.write('<ul id="navibar">');

   for (i=0; i < versions.length; i++) {
       document.write('<li class="wikilink">');
       document.write('<a href="' + proc_url + versions[i] + '">SciPy ' + versions[i] +  '</a>');
       document.write('</li>');
   }

   document.write('</ul>');
}

// range(start, end, step) 
// provides a range of integers or strings, depending on the type of input provided.
// If you input strings they will be converted to integers for the iteration logic to work.
// start: Int or string, first element to include in output
// end: Int or string, last element to include in output
// step: Int, amount and direction the array will be stepped through
var range = function(start, end, step) {
  var range = [];
  var typeOfStart = typeof start;
  var typeOfEnd = typeof end;

  if (typeOfStart == "undefined" || typeOfEnd == "undefined") {
    throw TypeError("Must pass start and end arguments.");
  } else if (typeOfStart != typeOfEnd) {
    throw TypeError("Start and end arguments must be of same type.");
  }
  if (typeOfStart == "string"){ 
    start = parseInt(start)
    end = parseInt(end)
  } 
  
  if (step === 0) {
    throw TypeError("Step cannot be zero.");
  }
  typeof step == "undefined" && (step = 1);

  if (end < start) {
    step = -step;
  }

  if (typeOfStart == "number") {
    while (step > 0 ? end >= start : end <= start) {
      range.push(start);
      start += step;
    }
  } else if (typeOfStart == "string") {
      while (step > 0 ? end >= start : end <= start) {
        range.push(start.toString());
        start += step;
      }
  } else {
    throw TypeError("Only number and string types are supported");
  }
  
  return range;
}
