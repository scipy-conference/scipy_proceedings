function proc_versions() {
   var versions = ['2011', '2010', '2009', '2008'];
   var proc_url = 'http://conference.scipy.org/proceedings/scipy';
   document.write('<ul id="navibar">');

   for (i=0; i < versions.length; i++) {
       document.write('<li class="wikilink">');
       document.write('<a href="' + proc_url + versions[i] + '">SciPy ' + versions[i] +  '</a>');
       document.write('</li>');
   }

   document.write('</ul>');
}
