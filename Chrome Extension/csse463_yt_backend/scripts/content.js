let lastUrl = location.href;

let currentCommentIndex = 0;
let comments = [];
let likeButtons = [];
let dislikeButtons = [];

function scrollToComment(index) {
    if (index >= 0 && index < comments.length) {
        comments[index].scrollIntoView({ behavior: "smooth", block: "start" });
        currentCommentIndex = index;
        console.log(`Scrolled to comment ${currentCommentIndex + 1}/${comments.length}`);
    } else if (comments.length === 0) {
        console.log("No comments found.");
    } else {
        console.log("Invalid comment index.");
    }
}

console.log("Content script loaded");

const observer = new MutationObserver(() => {
    const currentUrl = location.href;

    if (currentUrl !== lastUrl) {
        lastUrl = currentUrl;

        currentCommentIndex = 0;
        comments = Array.from(document.querySelectorAll('ytd-comment-thread-renderer.style-scope.ytd-item-section-renderer'));
        likeButtons = Array.from(document.querySelectorAll('ytd-toggle-button-renderer#like-button > yt-button-shape > button'));
        dislikeButtons = Array.from(document.querySelectorAll('ytd-toggle-button-renderer#dislike-button > yt-button-shape > button'));

        console.log(likeButtons)

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
        console.log('Ctrl+Alt+N key combination pressed!');
        comments = Array.from(document.querySelectorAll('ytd-comment-thread-renderer.style-scope.ytd-item-section-renderer'));
        if (currentCommentIndex < comments.length - 1) {
            scrollToComment(currentCommentIndex + 1);
        } else {
            document.querySelector('ytd-continuation-item-renderer.style-scope.ytd-item-section-renderer').scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
            console.log("Already at the last comment.");
        }
    } else if (event.ctrlKey && event.altKey && event.key === 'p') {
        // Previous Comment
        console.log('Ctrl+Alt+P key combination pressed!');
        if (currentCommentIndex > 0) {
            scrollToComment(currentCommentIndex - 1);
        } else {
            console.log("Already at the first comment.");
        }
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
        console.log('Ctrl+Alt+Q key combination pressed!');
        likeButtons = Array.from(document.querySelectorAll('ytd-toggle-button-renderer#like-button > yt-button-shape > button'));
        if (likeButtons.length > 0 && currentCommentIndex < likeButtons.length && currentCommentIndex >= 0) {
            likeButtons[currentCommentIndex].click();
            console.log(`Clicked like button at index ${currentCommentIndex}`);
        } else {
            console.error(`Invalid index or no like buttons found.`);
        }
    } else if (event.ctrlKey && event.altKey && event.key === 'w') {
        // Dislike Comment
        console.log('Ctrl+Alt+W key combination pressed!');
        dislikeButtons = Array.from(document.querySelectorAll('ytd-toggle-button-renderer#dislike-button > yt-button-shape > button'));
        if (dislikeButtons.length > 0 && currentCommentIndex < dislikeButtons.length && currentCommentIndex >= 0) {
            dislikeButtons[currentCommentIndex].click();
            console.log(`Clicked dislike button at index ${currentCommentIndex}`);
        } else {
            console.error(`Invalid index or no dislike buttons found.`);
        }
    }
});