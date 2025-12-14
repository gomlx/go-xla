package installer

import (
	"fmt"
	"sync"
	"time"

	"github.com/pkg/errors"
	"golang.org/x/term"
)

// Spinner is a utility to run an action while displaying a rotating text indicator
// and allowing the action to update the title text.
type Spinner struct {
	title  string
	action func(chan<- string)
}

// NewSpinner creates a new Spinner instance.
func NewSpinner() *Spinner {
	return &Spinner{}
}

// Title sets the initial title text for the spinner.
func (s *Spinner) Title(title string) *Spinner {
	s.title = title
	return s
}

// Action sets the function to be executed. The function receives a channel
// it can use to send real-time title updates.
func (s *Spinner) Action(action func(chan<- string)) *Spinner {
	s.action = action
	return s
}

const (
	eraseToEOL    = "\033[0K"
	hideCursor    = "\033[?25l"
	showCursor    = "\033[?25h"
	saveCursor    = "\0337"
	restoreCursor = "\0338"
)

var spinRunes = []rune("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")

// Run executes the action function and manages the spinner animation and title updates.
func (s *Spinner) Run() error {
	if s.action == nil {
		return errors.New("spinner action is not set")
	}

	// Constants for animation
	const interval = 200 * time.Millisecond

	// Channels for control and communication
	var wg sync.WaitGroup
	stopChan := make(chan struct{})
	// Channel for title updates, with some buffer space to avoid blocking.
	titleUpdateChan := make(chan string, 16)
	currentTitle := s.title // Start with the initial title

	// --- Spinner Animation Goroutine ---
	wg.Add(1)
	go func() {
		defer wg.Done()
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		spinCharsIdx := 0

		// Hide the cursor while the spinner is running
		fmt.Print(hideCursor)
		defer fmt.Print(showCursor)

		for {
			select {
			case <-ticker.C:
				// terminalWidth recalculated at every tick, since it may change mid-animation.
				terminalWidth, _, err := term.GetSize(0)
				if err != nil {
					terminalWidth = 80
				}

				// Animation tick: redraw with the current title
				spinRune := spinRunes[spinCharsIdx%len(spinRunes)]
				fmt.Printf("\r%c %s%s", spinRune, truncateToWidth(currentTitle, terminalWidth-3), eraseToEOL)
				spinCharsIdx++

			case newTitle := <-titleUpdateChan:
				currentTitle = newTitle

			case <-stopChan:
				// Cleanup and exit
				fmt.Printf("\r%s", eraseToEOL)
				return
			}
		}
	}()

	// --- Execute the user's action ---
	// The action receives the channel to post updates to
	s.action(titleUpdateChan)

	// Signal the spinner goroutine to stop and wait for it to finish
	close(stopChan)
	// Important: Do NOT close titleUpdateChan here. The animation goroutine
	// might still try to read from it before it fully shuts down.
	wg.Wait()

	return nil
}
