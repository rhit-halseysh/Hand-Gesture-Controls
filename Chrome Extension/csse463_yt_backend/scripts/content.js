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
        document.querySelector('button[aria-label*="comments" i], button[aria-label*="Comment" i]')?.click(); // ID from youtube source code
    } else if (event.ctrlKey && event.altKey && event.key === 'l') {
        // Like short
        console.log('Ctrl+Alt+L key combination pressed!');
        document.querySelector('button[aria-label*="like this video"]')?.click();
    } else if (event.ctrlKey && event.altKey && event.key === 'd') {
        // Dislike short
        console.log('Ctrl+Alt+D key combination pressed!');
        document.querySelector('button[aria-label*="Dislike this video"]')?.click();
    } else if (event.ctrlKey && event.altKey && event.key === 'r') {
        // Remove like/dislike short
        console.log('Ctrl+Alt+R key combination pressed!');
        document.querySelector('button[aria-label*="Dislike this video"]')?.click(); // Disliking twice removes like or dislike
        document.querySelector('button[aria-label*="Dislike this video"]')?.click();
    } else if (event.ctrlKey && event.altKey && event.key === 'x') {
        // Close comment section
        console.log('Ctrl+Alt+X key combination pressed!');
        document.querySelector('ytd-engagement-panel-section-list-renderer button[aria-label="Close"]')?.click();
    } else if (event.ctrlKey && event.altKey && event.key === 'n') {
        // Next Comment
        // TODO: Implement scrolling to next comment
        console.log('Ctrl+Alt+N key combination pressed!');
    } else if (event.ctrlKey && event.altKey && event.key === 'p') {
        // Previous Comment
        // TODO: Implement scrolling to previous comment
        console.log('Ctrl+Alt+P key combination pressed!');
    } else if (event.ctrlKey && event.altKey && event.key === 'b') {
        // Select comment box
        console.log('Ctrl+Alt+B key combination pressed!');
        document.querySelector('#placeholder-area')?.click();
    } else if (event.ctrlKey && event.altKey && event.key === 't') {
        // Previous short
        console.log('Ctrl+Alt+T key combination pressed!');
        document.querySelector('#navigation-button-up button')?.click();
    } else if (event.ctrlKey && event.altKey && event.key === 'y') {
        // Next short
        console.log('Ctrl+Alt+Y key combination pressed!');
        document.querySelector('#navigation-button-down button')?.click();
    } else if (event.ctrlKey && event.altKey && event.key === 'v') {
        // Post Comment
        console.log('Ctrl+Alt+V key combination pressed!');
        document.querySelector('ytd-button-renderer#submit-button button')?.click();
    } else if (event.ctrlKey && event.altKey && event.key === 'q') {
        // Like Comment
        // TODO: Implement liking a comment
        console.log('Ctrl+Alt+Q key combination pressed!');
    } else if (event.ctrlKey && event.altKey && event.key === 'w') {
        // Dislike Comment
        // TODO: Implement disliking a comment
        console.log('Ctrl+Alt+W key combination pressed!');
    }
});
