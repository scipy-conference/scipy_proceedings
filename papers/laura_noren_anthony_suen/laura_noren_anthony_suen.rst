:author: Laura Norén
:email: laura.noren@nyu.edu
:institution: New York University 

:author: Anthony Suen
:email: anthonysuen@berkeley.edu
:institution: University of California, Berkeley

------------------------------------------------------------------------------------
Equity, Scalability, and Sustainability of Data Science Infrastructure
------------------------------------------------------------------------------------

.. class:: abstract

We seek to understand the current state of equity, scalability, and sustainability of data science education infrastructure in both the U.S. and Canada. Our analysis of the technological, funding, and organizational structure of four types of institutions shows an increasing divergence in the ability of universities across the United States to provide students with accessible data science education infrastructure, primarily JupyterHub. We observe that generally liberal arts colleges, community colleges, and other institutions with limited IT staff and experience have greater difficulty setting up and maintaining JupyterHub, compared to well-funded private institutions or large public universities with a deep technical bench of IT staff. However by leveraging existing public-private partnerships and the experience of Canada’s national JupyterHub (Syzygy), the U.S. has an opportunity to provide a wider range of institutions and students access to JupyterHub. 


.. class:: keywords

data science education, Jupyter, Jupyterhub, higher education

Introduction
-----------------------

Data science education has experienced great demand over the past five years, with increasing numbers of undergraduate programs and majors being developed. This demand has fueled the growth of JupyterHubs, which create on-demand, cloud based Jupyter notebooks for students and researchers. Compared to local environments that run Jupyter, a JupyterHub provides many conveniences including not requiring any local installations and lower setup costs especially among various courses different courses, quicker access to course content since no downloading is required, grading, and computing flexibility, so that users even on Chromebooks or iPads are able to run Jupyter notebooks. 

Additional benefits include ability to quickly deploy customizations for different use cases, authentication, autograding, and providing campus-wide computing and storage. Overall, universities have found that utilizing JupyterHubs increases accessibility to data science tools, improves the scaling of data science and computing courses into many other domains, and provides a cohesive learning and research platform. 

However little was known about the barriers universities face when attempting to deploy JupyterHub, which has only been in use since 2015. 

This paper aims to understand how JupyterHub is affecting the equity, scalability, and sustainability of data science education by providing four cases studies of how JupyterHubs are being deployed in varying academic institutions across the United States and Canada. We will look at the barriers among these institutions to deploy, maintain, and grow JupyterHub from  technical staffing and financial perspectives. The four case studies include large and technical universities such as UC Berkeley, small liberal arts colleges, private universities with large endowments like Harvard, and the Canadian National JupyterHub Model. 

We conducted over 10 qualitative interviews with university faculty and IT staff from around the U.S. and Canada. We also reviewed documentation found on Github and websites of 20 institutions regarding their JupyterHub deployments. We structured our analysis by first trying to understand the institution’s educational goals and how it drives funding and decision/structure. We then delve into the infrastructural costs, capabilities, along with team size. We lastly measured educational impact, such as the number of students served and the number of classes. We conclude with a summary of the findings and potential ways to improve equity, scalability, and sustainability of current existing JupyterHub infrastructure. 


Case Study 1: UC Berkeley
------------------------------------

In Spring 2015, UC Berkeley became one of the first universities to adopt JupyterHub [1]_. Initially set up for 100 students in the new Foundations of Data Science Course, the JupyterHub instance quickly expanded as the data science initiative grew at Berkeley. As of Spring 2018, over 1,000 students take Data 8 each semester and around 3,000 students in Berkeley’s Data Science connector courses, modules, and upper division courses utilize the JupyterHub. An additional 45,000 students utilize the JupyterHub in Data 8's free online EdX version. 
				
UC Berkeley aims to serve large portions of its 30,000 undergraduates with data science tools, thus creating the motivation for it to build the largest JupyterHub deployments in the country. Its cross campus pedagogical vision is assisted by the presence of a large technical team, along with many members of the core Jupyter team. The Berkeley JupyterHub runs on the Kubernetes platform, which allows for easily scalable clusters that can support many thousands of users. Furthermore, Berkeley’s JupyterHub infrastructure, which subsists on cloud credits, is supported by long running industry relations and partnerships with cloud vendors like Microsoft and Google.
		
The UC Berkeley infrastructure team in charge of running Berkeley’s instance of JupyterHub, “Datahub”, consists of the Dean of the Division of Data Sciences, one tenured teaching faculty, one full-time staff member, ~10 postdocs and graduate students who can help troubleshoot–many of which are from the core Jupyter team–along with a large, technically proficient undergraduate support staff [2]_. Notably, a significant reason that allows UC Berkeley’s infrastructure to achieve its current level of robustness is that their dev/ops team is competent and is able to work closely with IT in various departments. 

Despite some of this success, UC Berkeley’s model faces sustainability challenges given the heavy reliance on undergraduates, graduate students and postdoc staff and donated computing credits from cloud vendors. Student and postdoc staff generally move on and have other priorities to advance their careers, leading to a lack of consistent support staff and a consequent lack of consistent expertise; they typically do not advance their careers by doing SysAdmin work. The reliance on free cloud credits is further not guaranteed forever and requires regular negotiations with public cloud vendors.

Nonetheless, Berkeley’s model benefits from its campus-wide scale, setting the ground for a large and diverse array of data science course to be setup with minimum infrastructure overhead [3]_. The infrastructure can also support very large courses, like quantitative gateway courses for many departments. Finally it provides a common suite of tools that are widely accessible, allowing students a productive and cohesive environment for both learning and research. 


Case Study 2: Small Liberal Arts Universities		
------------------------------------------------

The team interviewed several small liberal arts colleges to see how they utilized Jupyter in their data science or computer science curricula. We learned that lack of funding, insufficient technical know-how, limited relationships and experiences dealing with cloud vendors, and a shortage of time from busy instructors seem to be the major hurdles to a successfully deploying JupyterHub. 

At liberal arts colleges, deployments are usually designed for small classes consisting of ~20-30 students and deployed and maintained by one or two professors. There exists little IT help for the professor, as compared to the vast number of support staff at institutions like UC Berkeley. Some smaller institutions have even asked public institutions like UC Berkeley for support. The lack of proper guidance and departmental resources, along with overburdened faculty, often may dissuade efforts to set up JupyterHub all together. Generally, paying for such technology is also tough and ad hoc for smaller institutions. 

One of the exceptions is Bryn Mawr College; its Jupyterhub deployment currently hosts and allows access to a wide range of courses. Some courses such as *Introduction to Computing* (introductory computer science course) have migrated to the Jupyterhub environment, while new courses such as *Computing in Biology* have been introduced specifically utilizing Jupyter. Bryn Mawr has emphasized using Jupyterhub due to its accessibility for biology students who have limited experienced with programming, while also making it useful for CS students who are interested in biological applications for CS. The *Bio/CS 115: Computing Through Biology* course [4]_, which was developed based on the Jupyter environment, serves as an alternative CS intro course and a 2nd semester Biology intro cousrse. This option reduces the prerequisite barriers of entry to both domains and allows students to learn both subjects in an well-integrated manner, especially given the amount of intro courses that compete for their schedules. 


Case Study 3: Wealthy Private Universities	
------------------------------------------------

Compared to smaller liberal arts universities, the conditions are vastly different at well-funded private universities. Their ecosystem have plentiful IT professionals, and even if internal IT staff encounter limitations, rich private universities often pay third-party vendors to help deploy and maintain JupyterHubs and all related support infrastructure. Harvard has said that they “hired a firm to help us implement JupyterHub in Amazon AWS Cloud”. Compared to smaller liberal arts colleges, the experience is relatively free of frustration since the university covers all costs. Nonetheless, Harvard has noted that using JupyterHub has increased flexibility and hence decreased setup costs for both users and instructors, and has claimed that this solution is much more cost effective compared to traditional solutions. 

Most of the classes that have deployed JupyterHub are still relatively small, with most having 12-50 students. At Harvard, JupyterHub was deployed on AWS for two classes in the School of Engineering, which provided significant customization. The Signal Processing class used a Docker-based JupyterHub, where each user was provisioned with a docker container notebook. For the Decision Theory class, JupyterHub used a dedicated EC2 instance per user’s notebook, providing better scalability, reliability and cost efficiency [5]_. Its School of Engineering and Applied Science (SEAS) further announced in October 2017 for a schoolwide JupyterHub deployment [6]_. In addition to SEAS’s JupyterHub, the Harvard Medical School has its own JupyterHub deployment.

Instead of deploying and maintaining their own JupyterHubs, other universities have found success by contracting a third-party vendor to make their JupyterHub deployment experience completely hassle free. Vocareum [7]_, an example of one company that specializing in this space, helps to set up and manage environments like Jupyter and hosts labs for students to access. Currently, their data sciences lab is used by many wealthy private universities including Cornell, Columbia, and the University of Notre Dame. Others firms that provide similar services include CoCalc and Gryd.

Despite the hassle-free experiences, this model runs into major issues in replicability and scalability. Other universities generally have less experience with cloud computing or cannot rely on their university’s operating budget to support this type of teaching expense, especially if classes are relatively small (12-50 students). Furthermore, this model is very costly to scale as costs will grow with each instance, especially when factoring in deployment costs of contractors. This includes beyond just potential costs for the cloud provider, but also hiring outside consultants to setup JupyterHub. Setting up multiple individual hubs that nominally create the same type of service is wasteful, and long term costs could potentially sky rocket if there is no unified plan. One potential solution is to adopt Berkeley’s strategy of utilizing Kubernetes, which allows the JupyterHub to host thousands of students across many courses. 



Case Study 4: Canadian Federation (PIMS)	
------------------------------------------------

In 2017, an initiative in Canada led by the Pacific Institute of Mathematics and Sciences (PIMS) and Compute Canada started a new federated model for JupyterHub that provides access to numerous institutions across Canada [8]_. With data privacy laws removing option of cloud vendors, Syzygy is the largest federally funded JupyterHub and is utilized by more than 8,000 students in 15 different universities in Canada. Syzygy is run and supported by one full-time system network manager based at PIMS, who works with Compute Canada. The System Network manager is in charge of installations; any Canadian University can simply ask Syzygy for a JupyterHub and a new cluster will be set up. The system manager is paid for by Compute Canada, and further grants from the Canadian federal government ($4.5m) and Alberta ($1m) support professors and teachers. There is also time donation from professors at 10 different institutions. 
											
There are some potential bottlenecks with this model currently. For example, there is only one person conducting core management and operations for 15 different institutions. Some scaling issues also currently exist as any institution’s JupyterHub is at most able to handle ~2 classes of students concurrently (around 200-300 students). Nonetheless, this is a functional model in terms of scale and sustainability based on the number of universities involved, Canada’s population size, and strong central government support.

The leaders of the effort believe there are multiple benefits to the strategy. Firstly, it can accommodate small classes, modules, and also high schools across the country. Secondly, it allows instructors to focus on course development. Thirdly, it fosters better cross university collaboration by sharing experiences and course modules through a common network. 


Conclusion  - A Path Forward to a National Jupyterhub 				
--------------------------------------------------------------

While the grassroots efforts across the U.S. have sparked significant innovation in the realm of data science education infrastructure, it has also created a growing chasm of capabilities between institutions. Increasing training in statistics, computing, and data science is crucial to building the nation's STEM workforce, and such a national imperative requires a new model to scalably support many small institutions. 

Today, it is mainly large public or wealthy private universities in the U.S. can provide JupyterHub for large number of undergraduates. At smaller resource-constrained institutions, deploying a JupyterHub instance for a single class possesses nontrivial costs and may be daunting for one instructor or their university IT staff. Unfortunately, if there is no alternative way to access JupyterHub for data science education, smaller less well funded institutions and underrepresented communities cannot utilize JupyterHub.

When considering the future plans of Jupyterhub in higher data science education, we see four potential pathways: 

- **Status Quo** - Continue the current grassroots and uncoordinated JupyterHub deployments across institutions. Smaller or less resource rich institutions would likely continue to face existing barriers.

- **Institutional Grants** - Increasing foundational or governmental funding for individual universities to set up their JupyterHubs is another option. This can be done by allowing individual institutions to hire IT staff or paying third-party vendors to create a JupyterHub environment. Based on Berkeley’s and Harvard’s experiences, we’ve concluded that grants to hire staff to deploy Jupyterhub is non-scalable given the high costs of hiring IT staff with such specialized experience. On the other hand, funding third-party vendors like CoCalc, Gryd, Vocareum and public cloud providers like Google or Microsoft to help set up individual JupyterHubs is conceivable, but the fragmented nature of these transactions may end up being more costly than the coordinated national or regional models below. 

- **A National JupyterHub** - A national Jupyterhub would offer cost benefits such as utilizing existing federally funded national supercomputing centers. However a single national hub is difficult to realize due to high coordination costs with thousands of universities and the current political climate would not support adding more federal employees to manage this platform. As attractive as a national level JupyterHub may be, there are other scalable solutions that might be easier to coordinate and implement.  

- **Regional Hubs Models** - Establishing several regional hubs can reduce the burden of deployment and maintenance costs that individual universities experience today. For each regional network, by deploying a large Kubernetes cluster that can support many thousands of users, individual universities can then deploy their own JupyterHubs on the cluster. 

One proposal that scaffolds onto existing infrastructure the cloud credits from partners like Microsoft [9]_. The West Big Data Innovation Hub and UC Berkeley proposes to conduct a pilot program by setting up a Kubernetes cluster using Microsoft Azure for a small group of Western U.S. universities to pilot their JupyterHubs starting in the Summer of 2018. This will lower the administrative burden while providing a scalable infrastructure at a very low cost for many universities. Further integration of regional computing facilities at major research universities should be investigated. 

References
----------------------
.. [1] Kim, A. (2018, May 2). The Jupyterhub Journey: Starting Small and Scaling Up. Retrieved July 5, 2018, from https://data.berkeley.edu/news/jupyterhub-journey-starting-small-and-scaling
.. [2] Suen, A. (2018, March 15). People. Retrieved July 5, 2018, from https://data.berkeley.edu/about/people
.. [3] Kim, A. (2018, February 20). Modules: Data Made Accessible to Many. Retrieved July 5, 2018, from https://data.berkeley.edu/news/modules-data-made-accessible-many
.. [4] Shapiro, J. (2017, May 20). Computing Through Biology with Jupyter. Speech presented at Jupyter Day Philly, Philadelphia. Retrieved May 24, 2018, from https://github.com/BrynMawrCollege/TIDES/blob/master/JupyterDayPhilly/JAShapiro_JupyterDayPhilly_2017-05-19.pdf
.. [5] Harvard. (2018). cloudJHub. Retrieved May 24, 2018, from https://github.com/harvard/cloudJHub
.. [6] Ba, D. (2017, October 23). SEAS Computing and Academic Technology for FAS Launch JupyterHub Canvas Integration. Retrieved July 6, 2018, from https://atg.fas.harvard.edu/news/seas-computing-and-academic-technology-fas-launch-jupyterhub-canvas-integration
.. [7] DATA SCIENCES LAB @ VOCAREUM. (n.d.). Retrieved July 6, 2018, from https://www.vocareum.com/home/data-sciences-lab/
.. [8] Canadians Land on Jupyter. (2017, July 11). Retrieved May 24, 2018, from https://www.pims.math.ca/news/canadians-land-jupyter
.. [9] Mandava, V. (2017, June 8). NSF Big Data Innovation Hubs collaboration - looking back after one year - Microsoft Research. Retrieved May 24, 2018, from https://www.microsoft.com/en-us/research/blog/nsf-big-data-innovation-hubs-collaboration/
