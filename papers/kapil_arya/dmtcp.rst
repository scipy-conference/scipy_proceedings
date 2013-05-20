:author: Kapil Arya
:email: kapil@ccs.neu.edu
:institution: Northeastern University

:author: Gene Cooperman
:email: gene@ccs.neu.edu
:institution: Northeastern University

============================================
DMTCP: Bringing Checkpoint-Restart to Python
============================================

Introduction
============

DMTCP[1] is a mature user-space checkpoint-restart package.  One can
view checkpoint-restart as a generalization of pickling.  Instead of
saving an object to a file, one saves the entire Python session to a
file.  Checkpointing graphics in Python is also supported --- by
checkpointing a VNC session with Python running inside that session.

DMTCP is made accessible to Python as a Python module.  Hence, a
checkpoint is executed as "import dmtcp; dmtcp.checkpoint()".  This Python
module provides this and other functions to support the features of DMTCP.
The module for DMTCP functions equally well in IPython.

This DMTCP module implements a generalization of asaveWorkspace function,
which additionally supports graphics and the distributed processes of
IPython.  In addition, at least three novel uses of DMTCP for helping
debug Python are discussed.

1.  Fast/Slow Computation[3] --- Cython provides both traditional
    interpreted functions and compiled C functions.  Interpreted
    functions are slow, but correct.  Compiled functions are fast,
    but users sometimes declare incorrect C types, causing the
    compiled function silently return a wrong answer.  The idea
    of fast/slow computation is to run the compiled version on
    one computer node, while creating checkpoint images at regular
    intervals.  Separate computer nodes are used to check each interval
    in interpreted mode between checkpoints.

2.  FReD[2] --- a Fast Reversible Debugger that works closely with
    the Python pdb debugger, as well as other Python debuggers.

3.  Reverse Expression Watchpoint --- A bug occurred in the past.
    It is associated with the point in time when a certain 
    expression changed.  Bring the user back to a pdb session
    at the step before the bug occurred.

Background of DMTCP
===================

.. figure:: dmtcp-arch.eps
   :figwidth: 100%

   Architecture of DMTCP. :label:`dmtcp-arch`

DMTCP (Distributed MultiThreaded CheckPointing) [Ansel09]_ is a
transparent checkpoint-restart package with its roots going back eight
years [Rieker06]_.  It works completely in user space
and does not require any changes to the application or the operating
system.  DMTCP can be used to checkpoint a variety of user application
including Python.

Using DMTCP to checkpoint an application is as simple as executing the
following commands:

.. code-block:: sh

   dmtcp_checkpoint ./a.out
   dmtcp_command -c
   ./dmtcp_restart_script.sh

DMTCP automatically tracks all local and remote child processes and
their relationships.

As seen in Figure :ref:`dmtcp-arch` , a computation running under DMTCP
consists of a centralized coordinator process and several user
processes. The user processes may be local or distributed.  User
processes may communicate with each other using sockets, shared-memory,
pseudo-terminals, etc.  Further, each user process has a checkpoint
thread which communicates with the coordinator.

DMTCP Plugins
-------------
.. figure:: plugin-architecture-simple.eps
   :figwidth: 100%

   DMTCP Plugins. :label:`dmtcp-plugins`

DMTCP plugins are used to keep DMTCP modular. There is a separate plugin
for each operating system resource. Examples of plugins are pid plugin,
socket plugin, and file plugin. Plugins are responsible for
checkpointing and restoring the state of their corresponding resources.
The execution environment can change between checkpoint and restart. For
example, the computation might be restarted on a different computer
which has different file mount points, different network address, etc.
Plugins handle such changes in the execution environment by virtualizing
these aspects. Figure :ref:`dmtcp-plugins` shows the layout of DMTCP
plugins within the application.

DMTCP Coordinator
-----------------
DMTCP uses a stateless centralized process, the DMTCP coordinator, to
synchronize checkpoint and restart between distributed processes.
The user interacts with the  coordinator through the console to initiate
checkpoint, check status of the computation, kill the computation, etc.
It is also possible to run the coordinator as a daemon process, in which
case, the use may communicate with the coordinator using the command
``dmtcp_command``.

Checkpoint Thread
-----------------
The checkpoint thread waits for a checkpoint request from the DMTCP
coordinator.  On receiving the checkpoint request, the checkpoint thread
quiesces the user threads and creates the checkpoint image. To quiesce
user threads, it installs a signal handler for a dedicated POSIX signal
(by default, SIGUSR2).
Once the checkpoint image has been created, the user threads are allowed
to resume executing application code. Similarly, during restart, once the
process memory has been restored, the user threads can resume executing
application code.

Checkpoint
----------
On receiving the checkpoint request from the coordinator, the checkpoint
thread sends the checkpoint signal to all the user threads of the
process.  This quiesces the user threads by forcing them to block inside
a signal handler, defined by the DMTCP.  The checkpoint image is created
by writing all of user-space memory to a checkpoint image file. Each
process has its own checkpoint image.  Prior to checkpoint, each plugin
will have copied into user-space memory, any kernel state associated
with its concerns.  Examples of such concerns include network sockets,
files, and pseudo-terminals.  Once the checkpoint image has been
created, the checkpoint thread un-quiesces the user threads and they
resume executing application code.

At the time of checkpoint, all of user-space memory is written to a
checkpoint image file.  The user threads are then allowed to resume
execution.  Note that user-space memory includes the all of the run-time
libraries (libc, libpthread, etc.), which are also saved in the
checkpoint image.

In some cases, state outside the kernel must be saved.  For example, in
handling network sockets, data in flight must be saved.  This is done by
draining the network data by sending a *special cookie* through the
"send" end of each socket in one phase.  In a second phase, after a
global barrier, data is read from the "receive" end of each socket until
the special cookie is received. The in-flight data has now been copied
into user-space memory, and so will be included in the checkpoint image.
On restart, the network buffers are *refilled* by sending the in-flight
data back to the peer process, who then sends the data back into the
network.

Restart
-------
As the first step of restart phase, all memory areas of the process are
restored. Next, the user threads are recreated. The plugins then receive
the restart notification and restore their underlying resources,
translation tables etc.  Finally, the checkpoint thread un-quiesces the
user threads and the user threads resume executing application code.

DMTCP Python Integration
========================

DMTCP can checkpoint Python from the *outside* i.e. by treating
Python as a black box. To enable checkpointing, the Python interpreter
is launched in the following manner:

.. code-block:: sh

     $:> dmtcp_checkpoint python <args>

     $:> dmtcp_command -c

The command ``dmtcp_command`` can be used at any point to create a
checkpoint of the entire session.

DMTCP Module for Python
-----------------------
Checkpointing Python session or script from the outside doesn't provide
the user application with any mechanism for a finer grain control. A
typical use case arises in situation where the application wants to
checkpoint only at *safe points*. For example, if the application is
communicating with an external database server, checkpointing in the
middle of a transaction is undesired.
To solve this problem we present a DMTCP module for Python. This module
allows the application interact with
the DMTCP engine and enables the application to request a checkpoint at
pre-determined points in the code. In the following example, the
checkpoint request is made from within the application.

.. code-block:: python

   ...
   import dmtcp
   ...
   # Request a checkpoint if running under checkpoint
   # control
   dmtcp.checkpoint()
   # Checkpoint image has been created
   ...

It is also possible to do pre and post processing during checkpoint and
restart. The application can provide hooks that should be executed
during checkpoint and restart. A trivial way to execute pre and post
hooks during checkpoint and restart is exhibited in the following
example:

.. code-block:: python

   ...
   import dmtcp
   ...
   def my_ckpt(<args>):

       # Pre processing
       my_pre_ckpt_hook(<args>)
       ...
       # Create checkpoint
       dmtcp.checkpoint()
       # Checkpoint image has been created
       ...
       if dmtcp.isResume():
           # The process is resuming from a checkpoint
           my_resume_hook(<args>)
           ...
       else:
           # The process is restarting from a previous
           # checkpoint
           my_restart_hook(<args>)
           ...

       return
   ...

The function :code:`my_ckpt` can be defined in the application by the
user and can be called from within the user application at any point.

Extending DMTCP Module for Managing Sessions
--------------------------------------------
So far we have discussed the services provided by the DMTCP module to
interact with the DMTCP engine. These services can further extended to
provide the user with the concept of multiple sessions. A checkpointed
Python session is given a unique session id to distinguish it from other
sessions.  When running interactively, the user can view the list of
available checkpointed sessions.  The current session can be replaced by
any of the existing session using the session identifier.

The application can programmatically revert to an earlier session as
shown in the following example:

.. code-block:: python

   ...
   import dmtcp
   ...
   sessionId1 = dmtcp.checkpoint()
   ...
   sessionId2 = dmtcp.checkpoint()
   ...

   ...
   if <condition>:
       dmtcp.restore(sessionId2)
   else:
       dmtcp.restore(sessionId1)

Notice that only session id is used to restore to a previous session. It
is also possible to enhance the DMTCP module to pass extra arguments to
the restore function. Those extra arguments can be made available to the
:code:`dmtcp.isRestart()` path. The application can thus take a
different branch now instead of following the same route.

Save-Restore IPython Sessions
-----------------------------
To checkpoint an IPython session, one must consider the configuration
files. The configuration files are typically stored in user's home
directory. During restart, if the configuration files are missing, the
restarted computation may fail to continue.  Thus DMTCP, must checkpoint
and restore all the files that are required for proper restoration
of an IPython session.

Attempting to restore all configuration files during restart poses yet
another problem -- the existing configuration files might have newer
contents and overwriting them with copies from the checkpoint time may
not be desired by the user.  This may result in the user ending up losing important changes to those files.

One possible solution to handles this situation by taking snapshots of
the entire configuration directory along with the checkpoint image.
After restart, the IPython session should be made to use the
checkpointed copy of the configuration directory instead of the default
configuration directory.  This presents a significant challenge. The
IPython process remembers the old path, and the checkpointed copy of the
configuration directory has a different path. To handle this situation,
a DMTCP plugin is created for IPython. Whenever the IPython process
issues a system call to open a particular configuration file, the plugin
intercepts the system call and changes the file path to point to the
checkpointed copy.  The IPython process is unaware of the changes and
continues to work without any problems.

The session management capabilities of the DMTCP module can be further
extended to manage session for IPython. In case of IPython, each session
contains the configuration directory in addition to the checkpoint
image(s).

Save-Restore Parallel IPython Sessions
--------------------------------------

DMTCP is capable of checkpointing a distributed computations with
processes running on multiple nodes. It automatically checkpoints and
restores various kinds of inter process communication mechanisms such as
shared-memory, message queues, pseudo-ttys, pipes and network sockets. 

An IPython session involving a distributed computation running on a
cluster is checkpointed as a single unit. With DMTCP, it is possible to
restart the distributed processes in various manners. For example, for
debugging, it may be desirable to restart all the processes on a single
computer. In a different example, the processes may be restarted on a
different cluster altogether. Even further, the per node distribution
may be different from checkpoint time to accommodate changed nodes. 

Another use case involving parallel computations is to use
pre-initialized checkpoint images if multiple processes have a common long
initialization routine. Instead of having all processes go through the
same initialization, only one process is made to go through the
initialization and is checkpointed at the end of initialization.
Next, several processes are launched by restarting multiple copies of
this checkpoint image.

Fast/Slow Execution with Cython
===============================

A common problem for compiled versions of Python is how to check
whether the compiled computation is faithful to the interpreted
computation.  Compilation errors can occur if the compiled code
assumes a particular C type, and the computation violates that
assumption for a particular input.  Thus, one has to choose
between speed of computation and a guarantee that that the
compiled computation is faithful to the interpreted computation.

The core idea is to run the compiled code, while creating checkpoint
images at regular intervals.  A compiled computation interval is checked
by copying the two corresponding checkpoints (at the beginning and end of
the interval) to a separate computer node for checking.  The computation
is restarted from the first checkpoint image, on the checking node.
But when the computation is first restarted, the variables for all
user Python functions are set to the interpreted function object.
The interval of computation is then re-executed in interpreted mode
until the end of the computation interval.  The results at the end of
that interval can then be compared to the results at the end of the same
interval in compiled mode.

Checkpointing with graphics (inside vnc)
========================================
**FILL IN**

Reversible Debugging with FReD
==============================
While debugging a program, often the programmer over steps and has to
restart the debugging session. For example, while debugging a program,
if the programmer steps over (by issue :code:`next` command inside the debugger) a function :code:`f()` only to determine
that the bug is in function :code:`f()` itself, he is left with no
choice but to restart from the beginning.

*Reversible debugging* is the capability
to run the application backwards in time inside a debugger. If the
programmer detects that the problem is in function :code:`f()`, instead
of restarting from the beginning, he can issue a :code:`reverse-next`
command which takes it to the previous step. He can then issue
:code:`step` command to step into the function in order to find the
problem.

.. figure:: fred-arch-python.eps
   :figwidth: 200%

   Fast Reversible DeBugger. :label:`fred-arch`

FReD (Fast Reversible Debugger) [Arya12]_ is a reversible debugger based on
checkpoint-restart. FReD is implemented as a set of Python scipts and
uses DMTCP to create checkpoints during the
debugging session and keeps track of the debugging history. Figure
:ref:`fred-arch` shows the architecture of FReD.

A Simple UNDO Command
---------------------
The *UNDO* command reverses the effect of a previous debugger command
such as next, continue and finish. This is the most basic tool in
implementing a reversible debugger.

Getting the functionality of the UNDO command for debugging Python is
trivial.  A checkpoint is taken at the beginning of the debugging
session and a list of all debugging commands issued since the
checkpoint are recorded.

To execute UNDO command, the debugging session is restarted from the
checkpoint image, and the debugging commands are automatically
re-executed from the list excluding the last command.  This takes the
process back right before the debugger command was issued.

In longer debugging sessions checkpoints are taken at a frequent
interval to reduce the time spent in replaying the debugging history.

More complex reverse commands
-----------------------------
.. figure:: commands.eps
   :figwidth: 200%

   Reverse Commands. :label:`reverse-xxx`

Figure :ref:`reverse-xxx` shows some typical
debugging commands being executed in forward as well as backward
direction in time.

Suppose that the debugging history looks like :code:`[next,next]`
i.e. the user issued two :code:`next` commands. Further, the second next
command stepped over a function :code:`f()`.
Suppose we take checkpoints before each of these commands.
Issuing a :code:`reverse-next` command is easy. Just restart from the
last checkpoint image. However, if the command issued was
:code:`reverse-step`, a simple undo may not work. In this case, the
desired behavior is to take the debugger to the last statement of
the function :code:`f()`. In such situations we need to decompose the
last command [Visan11]_ into a series of commands. At the end of
this decomposition, the last command in the history is a :code:`step`.
At this point, the
history may look like: :code:`[next,step,next, ...,next,step]`. At this
point, the process is restarted from the last checkpoint and the
debugging history is executed excluding the last :code:`step` command.

A typical debuggin session in FRed with Python
----------------------------------------------

.. code-block:: python
   :linenos:

   $:> fredapp.py -mpdb python a.py
   (Pdb) break main
   (Pdb) run
   (Pdb) fred checkpoint
   (Pdb) break 6
   (Pdb) continue
   (Pdb) fred-history
   [break 6, continue]
   (Pdb) fred-reverse-next
   (Pdb) fred-history
    [break 7, next, next, next, next, next, next, next,
     next, next, next, step, next, next, next, where]

Reverse Expression Watchpoints
------------------------------

The *reverse expression watchpoint* automatically finds the location of
the fault for a given expression in the history of the program
execution.  It brings the user directly to a statement (one that is not
a function call) at which the expression is correct, but executing the
statement will cause the expression to become incorrect.

.. figure:: rw-new.eps
   :figwidth: 200%

   Reverse Expression Watchpoint. :label:`reverse-watch`

Figure :ref:`reverse-watch` provides a simple example.  Assume that a
bug occurs whenever a linked list has length longer than one million.
So an expression :code:`linked_list.len() <= 1000000` is assumed to be
true throughout.  Assume that it is too expensive to frequently compute
the length of the linked list, since this would require :math:`O(n^2)`
time in what would otherwise be a :math:`O(n)` time algorithm.  (A more
sophisticated example might consider a bug in an otherwise
duplicate-free linked list or an otherwise cycle-free graph.  But the
current example is chosen for ease of illustrating the ideas.)

If the length of the linked list is less than or equal to one million,
call the expression "good".  If the length of the linked list is greater
than one million, call the expression "bad".  A "bug" is defined as a
transition from "good" to "bad".  There may be more than one such
transition or bug over the process lifetime.  Our goal is simply to find
any one occurrence of the bug.

The core of a reverse expression watchpoint is a binary search.  In
Figure :ref:`reverse-watch`, assume a checkpoint was taken near the
beginning of the time interval.  So, we can revert to any point in the
illustrated time interval by restarting from the checkpoint image and
re-executing the history of debugging commands until the desired point
in time.

Since the expression is "good" at the beginning of Figure
:ref:`reverse-watch` and it is "bad" at the end of that figure, there
must exist a buggy statement --- a statement exhibiting the transition
from "good" to "bad".  A standard binary search algorithm converges to
some instance in which the next statement transitions from "good" to
"bad".  By definition, FReD has found the statement with the bug.  This
represents success.

If implemented naively, this binary search requires that some statements
may need to be re-executed up to :math:`\log_2 N` times.  However, FReD
can also create intermediate checkpoints.  In the worst case, one can
form a checkpoint at each phase of the binary search.  In that case, no
particular sub-interval over the time period needs to be executed more
than twice.






References
==========

.. [Ansel09] Jason Ansel, Kapil Arya, and Gene Cooperman.
           *DMTCP: Transparent Checkpointing for Cluster Computations
           and the Desktop*,
           23rd IEEE International Symposium on Parallel and Distributed
           Processing (IPDPS-09), 1-12, 2009
           http://dmtcp.sourceforge.net/.

.. [Arya12] Kapil Arya, Tyler Denniston, Ana Maria Visan, and Gene
           Cooperman.
           *FReD: Automated Debugging via Binary Search through a
           Process Lifetime*,
           http://arxiv.org/abs/1212.5204.

.. [Rieker06] Michael Rieker, Jason Ansel, and Gene Cooperman.
           *Transparent User-Level Checkpointing for the Native POSIX
           Thread Library for Linux*,
           Proceeding of PDPTA-06, 492-498, 2006.

.. [Visan11] Ana-Maria Visan, Kapil Arya, Gene Cooperman, and Tyler
           Denniston.
           *URDB: A Universal Reversible Debugger Based on Decomposing
           Debugging Histories*,
           In Proc. of 6th Workshop on Programming Languages and Operating
           Systems (PLOS'2011) (part of Proc. of 23rd ACM SOSP), 2011.
