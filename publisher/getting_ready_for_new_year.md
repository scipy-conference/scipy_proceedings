# Getting ready for the new year's proceedings

We're going to assume that you are working in year *x* to set up the proceedings
system for SciPy *x*. 

This workflow requires a few steps. We will list those steps here, but if you
don't know what some of the words mean, we highly recommend that you read 
[the problem section](#What-problem-are-we-solving).

# Workflow

All development work should have been occurring on the `dev` branch of the repo.

1. Merge `dev` into `master`
1. Create new branch for year `x` based off of the new `master`
1. Update the default branch
   - ![Update default branch](./images/update_default_branch.png)
1. (potential step) If you have a new server location, you need to update the webhook
1. Update the docs to reflect all of the above changes
   - that includes:
       - `scipy_proceedings/README.md`
       - `scipy_proceedings/publisher/README.md`
       - `scipy_proceedings/publisher/getting_ready_for_new_year.md`
