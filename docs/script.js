// Highlights the active section link in the top nav while scrolling.
(function () {
  var sections = Array.prototype.slice.call(document.querySelectorAll("section[id]"));
  var links = Array.prototype.slice.call(document.querySelectorAll(".navlinks a[href^='#']"));

  if (!sections.length || !links.length) return;

  var linkById = {};
  links.forEach(function (link) {
    linkById[link.getAttribute("href").slice(1)] = link;
  });

  var observer = new IntersectionObserver(
    function (entries) {
      entries.forEach(function (entry) {
        var link = linkById[entry.target.id];
        if (!link) return;
        if (entry.isIntersecting) {
          links.forEach(function (l) { l.classList.remove("active"); });
          link.classList.add("active");
        }
      });
    },
    { rootMargin: "-40% 0px -50% 0px", threshold: 0 }
  );

  sections.forEach(function (section) { observer.observe(section); });
})();
