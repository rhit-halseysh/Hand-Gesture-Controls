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

document.addEventListener('keydown', function (event) {
    if (event.ctrlKey && event.altKey && event.key === 'c') {
        // Open comments section
        console.log('Ctrl+Alt+C key combination pressed!');
        const targetButton = document.querySelector('button[aria-label*="comments" i], button[aria-label*="Comment" i]'); // ID from youtube source code
        targetButton?.click();
    } else if (event.ctrlKey && event.altKey && event.key === 'l') {
        // Like short
        console.log('Ctrl+Alt+L key combination pressed!');
    } else if (event.ctrlKey && event.altKey && event.key === 'd') {
        // Dislike short
        console.log('Ctrl+Alt+D key combination pressed!');
    } else if (event.ctrlKey && event.altKey && event.key === 'x') {
        // Close comment section
        console.log('Ctrl+Alt+X key combination pressed!');
    } else if (event.ctrlKey && event.altKey && event.key === 'n') {
        // Next Comment
        console.log('Ctrl+Alt+N key combination pressed!');
    } else if (event.ctrlKey && event.altKey && event.key === 'p') {
        // Previous Comment
        console.log('Ctrl+Alt+P key combination pressed!');
    } else if (event.ctrlKey && event.altKey && event.key === 'b') {
        // Select comment box
        console.log('Ctrl+Alt+B key combination pressed!');
    } else if (event.ctrlKey && event.altKey && event.key === 't') {
        // Previous short
        console.log('Ctrl+Alt+T key combination pressed!');
    } else if (event.ctrlKey && event.altKey && event.key === 'y') {
        // Next short
        console.log('Ctrl+Alt+Y key combination pressed!');
    } else if (event.ctrlKey && event.altKey && event.key === 'v') {
        // Post Comment
        console.log('Ctrl+Alt+V key combination pressed!');
    } else if (event.ctrlKey && event.altKey && event.key === 'q') {
        // Like Comment
        console.log('Ctrl+Alt+Q key combination pressed!');
    } else if (event.ctrlKey && event.altKey && event.key === 'w') {
        // Dislike Comment
        console.log('Ctrl+Alt+W key combination pressed!');
    }
});
