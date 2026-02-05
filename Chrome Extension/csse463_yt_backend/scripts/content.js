let lastUrl = location.href;

console.log("Content script loaded");

const observer = new MutationObserver(() => {
    const currentUrl = location.href;

    if (currentUrl !== lastUrl) {
        lastUrl = currentUrl;

        if (currentUrl.includes("/shorts/")) {
            console.log("Test worked");
        }
    }
});

observer.observe(document.body, {
    childList: true,
    subtree: true
});
