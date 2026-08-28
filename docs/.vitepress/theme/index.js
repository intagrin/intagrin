import DefaultTheme from 'vitepress/theme'
import './custom.css'

// enhanceApp runs during Vue app setup, before the current route's Markdown content is actually
// mounted into the DOM — a plain `document.readyState === 'complete'` (or even 'load') check
// fires too early and finds zero matching elements, silently never wiring anything up on a fresh
// page load (VitePress's router only calls onAfterRouteChange for *subsequent* client-side
// navigations, plus once more shortly after the initial mount). Polling briefly for elements to
// exist sidesteps needing to know the exact mount timing for either case. `run` is only called
// once per (selector, route) — repeated polls/route-change echoes for the same mount are no-ops.
function whenMounted(selector, run) {
  let done = false
  const tryRun = () => {
    if (done) return false
    const targets = document.querySelectorAll(selector)
    if (!targets.length) return false
    done = true
    run([...targets])
    return true
  }
  if (tryRun()) return

  let attempts = 0
  const poll = setInterval(() => {
    attempts += 1
    if (tryRun() || attempts > 40) clearInterval(poll) // ~4s, then give up quietly
  }, 100)
}

// Fade-and-rise reveal for the landing page's below-the-fold sections as they scroll into view —
// a generic, widely-used technique (IntersectionObserver toggling a class), not lifted from any
// particular site. Scoped to .landing-section only, so it never touches ordinary doc pages or the
// hero/feature grid, which stay visible immediately on load as before.
function setUpScrollReveal() {
  try {
    if (!('IntersectionObserver' in window)) {
      document.querySelectorAll('.landing-section').forEach((el) => el.classList.add('is-visible'))
      return
    }
    const observer = new IntersectionObserver(
      (entries, obs) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('is-visible')
            obs.unobserve(entry.target)
          }
        })
      },
      { threshold: 0.1, rootMargin: '0px 0px -40px 0px' }
    )
    whenMounted('.landing-section', (targets) => targets.forEach((el) => observer.observe(el)))
  } catch (e) {
    // Best-effort polish, never allowed to leave real content invisible.
    document.querySelectorAll('.landing-section').forEach((el) => el.classList.add('is-visible'))
  }
}

function currentTheme() {
  return document.documentElement.classList.contains('dark') ? 'dark' : 'light'
}

// Loads only the theme-matching pair of <source> elements — never both light and dark at once,
// which would double the video bandwidth/decode cost for a clip the visitor only ever sees one
// half of. video.dataset.loadedTheme tracks which pair is currently attached so a theme toggle
// that fires the MutationObserver below without an actual light/dark change (attribute churn
// unrelated to .dark, or the same value re-observed) is a no-op instead of an unnecessary reload.
function loadThemedSource(video, theme) {
  if (video.dataset.loadedTheme === theme) return
  const webm = video.dataset[theme === 'dark' ? 'webmDark' : 'webmLight']
  const mp4 = video.dataset[theme === 'dark' ? 'mp4Dark' : 'mp4Light']
  if (!webm || !mp4) return

  video.innerHTML = ''
  const sourceWebm = document.createElement('source')
  sourceWebm.src = webm
  sourceWebm.type = 'video/webm'
  const sourceMp4 = document.createElement('source')
  sourceMp4.src = mp4
  sourceMp4.type = 'video/mp4'
  video.append(sourceWebm, sourceMp4)
  video.dataset.loadedTheme = theme
  video.load()

  // The video is silent (no audio track) and purely decorative/informative looping motion —
  // exactly what prefers-reduced-motion exists for. autoplay/loop are HTML attributes CSS can't
  // override, so this leaves it paused on its first frame (a static image) for anyone who's told
  // their OS they want less motion, instead of an unskippable moving background.
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    video.pause()
  } else {
    // Autoplay can still be blocked by some browser policies after a JS-driven load(); that's a
    // silent no-op degrading to a static first frame, not an error worth surfacing.
    video.play().catch(() => {})
  }
}

function setUpStoryVideo() {
  try {
    whenMounted('.story-video', (targets) => {
      targets.forEach((video) => loadThemedSource(video, currentTheme()))

      if (!('MutationObserver' in window)) return
      const observer = new MutationObserver(() => {
        targets.forEach((video) => loadThemedSource(video, currentTheme()))
      })
      observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] })
    })
  } catch (e) {
    // Best-effort polish, never allowed to block the page over a video failing to wire up.
  }
}

export default {
  ...DefaultTheme,
  enhanceApp({ router }) {
    if (typeof window === 'undefined') return // skip during SSG build
    router.onAfterRouteChange = () => {
      setUpScrollReveal()
      setUpStoryVideo()
    }
    setUpScrollReveal()
    setUpStoryVideo()
  }
}
