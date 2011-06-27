:author: Chuck Thier
:email: cthier@gmail.com
:institution: Rackspace

------------------------------------------------------------------------
Lessons Learned Building a Scalable Distributed Storage System in Python
------------------------------------------------------------------------

.. class:: abstract


.. class:: keywords

   python, storage, distributed


What is Swift?
--------------

* distributed object storage
* Rackspace CloudFiles
* OpenStack object storage


The Challenge
-------------

* 100 petabytes of storage
* 100 billion object
* 100 gigabit/sec throughput
* 100 thousand requests per second


"At Scale, Everything Breaks"
-----------------------------

* 100 PB = 60,000+ Hard drives = 2500+ Storage Nodes
* Multiple failures daily even at smaller scales
* Handling failure is priority #1


Handling Failure
----------------

* fail fast
  * timeouts
  * expect 100-continue
* fail gracefully
  * handoff
  * error limiting
* recover from failure
  * replication


Enemy #1.0: Network I/O
-----------------------

* saturating GigE is easy, 10GigE needs a lot of CPU
* SSL on 10GigE
* python threading is problematic
* twisted doesn't fit my brain
* eventlet to the rescue


Enemy #1.1: Disk I/O
--------------------

* RAID 5/6 performance
  * random I/O is worst case scenario
  * several week rebuild times on failure
  * 48TB filesystems are problematic
* filesystems
  * fsync changes the game a bit
  * performance degradation over time
  * directory listings
* async file I/O is a pipe dream
* chunking file opearations is "good enough"


Questions?
----------

* Chuck Thier (cthier@gmail.com)
* Swift (http://launchpad.net/swift)
* Openstack (http://openstack.org)
