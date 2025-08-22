---
title: Appendix
---

(appendix-history)=

## History & Background

In this section, we share a brief history of the Jupyter Book project with the goal of providing context for the path that led to JB2, as well as to share credit and acknowledgement for the hundreds of contributors that have helped the project over the years.

JB2 is the third significant re-write of the Jupyter Book stack in the past decade, each of which involved hundreds of collaborators and users. The original version was built in 2018 as a loose collection of template files utilizing nbconvert for execution and Jekyll for rendering notebooks into websites, originally built for the [Data 8 textbook](http://inferentialthinking.com).

In 2020, Jupyter Book was re-written and formally released as Jupyter Book 1 (JB1). This was built on the Sphinx documentation generator, and resulted in the creation of the [MyST Markdown syntax and parser](http://myst-parser.readthedocs.io) for Sphinx. This work was largely funded by the [Executable Books Project](http://executablebooks.org), a funded by the Sloan Foundation ([Grant #9231](https://sloan.org/grant-detail/9231)), Jupyter Meets the Earth ([NSF grant #1928406](https://www.nsf.gov/awardsearch/showAward?AWD_ID=1928406)). While the Executable Books Project has completed, maintenance of the Jupyter Book 1 stack continues to this day.

Between 2020 and 2023, the Executable Books Project began a parallel collaboration with the company [Curvenote](http://curvenote.com) to explore a standards- and web-based workflow for the MyST Markup language (funded in part by Alberta Innovates and the Stanford Doerr School of Sustainability). As a result, Curvenote integrated their document engine as an upstream project in the `executablebooks/`organization and the project developed it together from there. This became the starting point for the `mystmd` stack described in this article.

From 2023 onward, the wider community has invested in `mystmd` and improved the capabilities of the command line tool, parsing capabilities, templates, and web themes. The next version of JupyterBook, Jupyter Book 2 (JB2), is built on top of this engine. In 2024, the project moved from being an independent organization to being incorporated as an official Jupyter sub-project, standardizing on using and stewarding the MyST document engine ([See #123](https://github.com/jupyter/enhancement-proposals/pull/123)).

:::{figure} history.png
:label: fig:history

Major phases of Jupyter Book development.
:::

### A note on migrating from Jupyter Book 1

A key goal of Jupyter Book 2 was to leverage the design and standards from Jupyter Book 1 and the Sphinx stack in order to facilitate the upgrade process. As a result JB2 leverages the same MyST Markdown syntax as JB1 with minimal disruption. It also aims to expose a key subset of the extension points that were available in JB1 and Sphinx (for example, roles, directives, custom transforms, etc). While there is a subset of functionality that is still unique to Sphinx, the JB2 team is focusing their efforts on developing key missing functionality to narrow this gap. See the [Jupyter Book 2 migration guide](https://next.jupyterbook.org/upgrade/) for more information.

(appendix-plugins)=

## Examples of plugins

**Directives:** Here is an example of a directive logic that generates an `{image}` node in the AST by pulling a random image from [picsum](https://picsum.photos/). Note how we define arguments and options, similar to how a function would be defined in a programming language.

```javascript
const picsumDirective = {
  name: 'picsum',
  doc: 'An example directive for showing a nice random image at a custom size.',
  alias: ['random-pic'],
  arg: {
    type: String,
    doc: 'The ID of the image to use, e.g. 1',
  },
  options: {
    size: { type: String, doc: 'Size of the image, for example, `500x200`.' },
  },
  run(data) {
    // Parse size
    const match = (data.options?.size ?? '').match(/^(\d+)(?:x(\d+))?$/);
    let sizeQuery = '200/200';
    if (match) {
      const first = match[1];
      const second = match[2];
      sizeQuery = second ? `${first}/${second}` : first;
    }

    const idQuery = data.arg ? `id/${data.arg}/` : '';
    const url = `https://picsum.photos/${idQuery}${sizeQuery}`;
    const img = { type: 'image', url };
    return [img];
  },
};

const plugin = { name: 'Lorem Picsum Images', directives: [picsumDirective] };
export default plugin;
```

**Transforms:** Here is an example of transform logic that parses the MyST AST and replaces **strong** styling with _emphasis_ styling. It operates on all `node` objects in the document using the `selectAll` utility function.

```javascript
const plugin = {
  name: 'Strong to emphasis',
  transforms: [
    {
      name: 'transform-typography',
      doc: 'An example transform that rewrites bold text as text with emphasis.',
      stage: 'document',
      plugin: (_, utils) => (node) => {
        utils.selectAll('strong', node).forEach((strongNode) => {
          const childTextNodes = utils.selectAll('text', strongNode);
          const childText = childTextNodes.map((child) => child.value).join('');
          if (childText === 'special bold text') {
            strongNode['type'] = 'span';
            strongNode['style'] = {
              background: '-webkit-linear-gradient(20deg, #09009f, #E743D9)',
              '-webkit-background-clip': 'text',
              '-webkit-text-fill-color': 'transparent',
            };
          }
        });
      },
    },
  ],
};

export default plugin;
```

(appendix-myst-xref)=

## Example myst.xref.json

Here’s an example of references in a `myst.xref.json` file from the [MyST Guide](http://mystmd.org/guide/myst.xref.json), note both page references and “identifier” references (e.g., labels attached to a figure or header):

```json
references:
{ kind: "page", data: "/index.json", url: "/" }
{ identifier: "cool-myst-features", kind: "heading", data: "/index.json", … }
{ identifier: "quickstart-tutorials", kind: "heading", data: "/index.json", … }
{ identifier: "cite-mystmd", kind: "heading", data: "/index.json", … }
{ identifier: "project-goals", kind: "heading", data: "/index.json", … }
{ kind: "page", data: "/installing.json", url: "/installing" }
{ identifier: "installing-myst-tabs", kind: "tabSet", data: "/installing.json", … }
```

(appendix-composable)=

## Composable Configuration

MyST configuration files can be composed with one another using the `extends:` keyword. This allows configuration to be split across multiple files, or even downloaded from a remote source via the web. For example, an author might put their author affiliation information in a dedicated `authors.yml` file:

```yaml
version: 1
project:
  contributors:
    - id: person_a
      name: First Last
      email: person_a@org_a.org
      orcid: XXXX-XXXX-XXXX-XXXX
      github: person_a
      affiliations:
        - id: org_a
        - id: org_b
    - id: person_b
      name: First Last
      email: person_b@org_b.com
      orcid: XXXX-XXXX-XXXX-XXXX
      github: person_b
      affiliations:
        - id: org_b
  affiliations:
    - id: org_a
      name: Organization A Incorporated
    - id: org_b
      name: Org B Inc.
```

And then re-use this affiliation information across multiple Jupyter Books or MyST Projects by “extending” the configuration like so:

```yaml
version: 1
extends:
  # Our local authors file
  - authors.yml
  # A remote configuration file
  - https://raw.githubusercontent.com/myorg/myrepo/refs/heads/main/funding.yaml
# Any other MyST configuration here
site: ...
project: ...
```

By allowing configuration to be split across multiple files and re-used easily, communities can reduce the duplication and outdated content associated with having a copy of the same information in multiple places. This is particularly useful for communities that need centralized databases of community-wide information (like author information) that they wish to re-use in multiple places. It’s also useful to standardize community-specific configuration like branding or links in the site navigation bar.

(appendix-jlab-myst)=

## Authoring in computational environments with `jupyterlab-myst`

A major motivation for rebuilding Jupyter Book on the MyST ecosystem is to enable a truly **web-native runtime model**. The MyST stack is built in JavaScript, and leverages the `unist` syntax tree ecosystem, a specification for ASTs used in content transformation pipelines, making it directly compatible with modern browser environments and frontend frameworks like React, as well as a host of other plugins and transformations (e.g., `unified-latex`). This also allows us to develop MyST-native authoring applications, such as `jupyterlab-myst`, a plugin that brings MyST rendering capabilities directly into JupyterLab.

With `jupyterlab-myst`, authors can preview how their markdown content elements will appear in the rendered book—including directives, equations, citations, and even code outputs—without leaving the notebook environment. This tight feedback loop reduces friction during authoring, while preserving a single source of truth between exploratory notebooks and published outputs. The plugin is part of our vision to unify **authoring, execution, and publishing** in one environment. Rather than using separate tools for writing, developing, and rendering, the MyST ecosystem will allow interfaces like JupyterLab as a single full-featured writing and publishing environment. This lowers the barrier to creating high-quality computational narratives, especially for teams already working with Jupyter tools for their day-to-day research and teaching.

:::{figure} jupyterlab-myst
:label: fig:jlab-myst
Using `jupyterlab-myst` to edit MyST content with embedded widgets and interactivity in the markdown cells.
:::
