function proc_versions() {
   var versions = ['2023', '2022', '2020', '2019', '2018', '2017', '2016', '2015', '2014', '2013', '2012', '2011', '2010', '2009', '2008'];
   var proc_url = 'https://conference.scipy.org/proceedings/scipy';
   document.write('<ul id="navibar">');

   for (i=0; i < versions.length; i++) {
       document.write('<li class="wikilink">');
       document.write('<a href="' + proc_url + versions[i] + '">SciPy ' + versions[i] +  '</a>');
       document.write('</li>');
   }

   document.write('</ul>');
}
