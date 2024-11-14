# Decisions & Dragons Website Repo
This is the repository for the [Decisions & Dragons](https://www.decisionsanddragons.com/) website: a website for answering reinforcement learning questions.

## Requesting an answer
If you have a question that you think I should answer on the site, you can reach out to me through any of 
[my social links at the top of the about page](https://www.decisionsanddragons.com/about/), or you can submit your question via Github issues. 
Please note that I selectively publish questions and answers that I think will provide broader insights, so I might not publish your question. 
However, even if I don't publish your question, I'll do my best to respond personally!

## Corrections
I am running this site by myself. On my own, I tend to make a lot of typos. If you find any, please let me know either by filing a Github issue or a pull request.
I would not reccomend making pull requests for larger changes because I would like to keep a consistent voice for the site. However, I welcome feedback
through Github issues. Similarly, if you think there is a technical error or oversimplification, let me know via Github issues.

I am keeping a [Corrections page](https://www.decisionsanddragons.com/corrections/) on the site where I list user-submitted corrections.

## Development
The site is built using Hugo and is deployed via github pages. The actual content is written in markdown found in the `content` directory. I am using a custom theme
which I am just calling the "Decisions & Dragons" theme. The markdown and theme support the use of mathjax equations using `$inline latex$` and `$$block latex$$` syntax. 
By default, content pages will have a table of contents (ToC) side bar generated for all headers (except h1 headers which should only be used for the page title). However,
the ToC sidebar is not rendered on mobile browsers to conserve space. You can also disable the ToC sidebar for a page by setting `layout = "single_no_toc"` in the front 
matter of the markdown. You can use code and language annotated code blocks in the markdown and you can use footnotes. Finally,
special hugo codes can be included in the markdown for figures. It follows the form:
```
{{< figure src="img/path.png" title="A figure title/caption" >}}
```

### Building
You must have [Hugo](https://gohugo.io/) installed on your system. From the repository root, run:
```bash
hugo server
```
which will both compile the website (creating a `public` directory containing the static html) and start a local webserver hosting it.

Deployment is handled by Github actions whenever `main` is updated on Github.
