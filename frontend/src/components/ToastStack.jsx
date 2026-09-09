const ICONS = {
    success: "✓",
    error: "!",
    info: "i",
};

function ToastStack({ toasts, onDismiss }) {

    if (!toasts.length) {
        return null;
    }

    return (

        <div className="toast-stack" role="status" aria-live="polite">

            {toasts.map((toast) => (

                <div key={toast.id} className={`toast toast-${toast.type}`}>

                    <span className="toast-icon" aria-hidden="true">
                        {ICONS[toast.type] || ICONS.info}
                    </span>

                    <span className="toast-message">{toast.text}</span>

                    <button
                        type="button"
                        className="toast-close"
                        onClick={() => onDismiss(toast.id)}
                        aria-label="Dismiss notification"
                    >
                        ×
                    </button>

                </div>

            ))}

        </div>

    );

}

export default ToastStack;
