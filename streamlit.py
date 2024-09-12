import random

import streamlit as st
from lib import (
    Add,
    AdditionCrosses10Boundary,
    AdditionCrosses100Boundary,
    Equal,
    IsDivisibleBy,
    IsGreaterThan,
    IsLessThan,
    Literal,
    Variable,
    expression_string,
    find_bindings,
)
from streamlit.components.v1 import html

x = Variable("x")
y = Variable("y")
z = Variable("z")

vars = {
    "x": x,
    "y": y,
    "z": z,
}

lhs = Add(x, y)
rhs = z

constraints = [
    # IsGreaterThan(y, x),
    IsGreaterThan(x, Literal(100)),
    IsLessThan(y, Literal(10)),
    AdditionCrosses10Boundary(x, y),
    # IsLessThan(x, Literal(20)),
    # IsLessThan(y, Literal(20)),
    Equal(lhs, rhs),
]
constraints = [
    # IsGreaterThan(y, x),
    IsGreaterThan(x, Literal(100)),
    IsLessThan(y, Literal(100)),
    IsDivisibleBy(x, Literal(10)),  # Remove to make harder
    IsDivisibleBy(y, Literal(10)),
    AdditionCrosses100Boundary(x, y),
    # IsLessThan(x, Literal(20)),
    # IsLessThan(y, Literal(20)),
    Equal(lhs, rhs),
]
domains = {v.name: list(range(0, 1000)) for v in [x, y, z]}
bindings = find_bindings(["x", "y", "z"], domains, constraints)


def get_next_problem():
    return next(bindings)


def main():
    st.set_page_config(
        page_title="Baloney Plays Football", page_icon="⚽", layout="wide"
    )

    if "problem" not in st.session_state:
        st.session_state.problem = get_next_problem()
        st.session_state.hold_out = random.choice(
            [
                y,
                z,
                z,
                z,
                z,
                z,
            ]
        ).name
        st.session_state.correct_answer = False

    if st.session_state.correct_answer:
        display_text = (
            expression_string(
                lhs, st.session_state.problem, underline=vars[st.session_state.hold_out]
            )
            + " = "
            + expression_string(
                rhs, st.session_state.problem, underline=vars[st.session_state.hold_out]
            )
        )
    else:
        display_text = (
            expression_string(
                lhs, st.session_state.problem, hold_out=vars[st.session_state.hold_out]
            )
            + " = "
            + expression_string(
                rhs, st.session_state.problem, hold_out=vars[st.session_state.hold_out]
            )
        )

    st.markdown(
        f"<div style='font-size: 150px; text-align: center;'>{display_text}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br/>" * 2, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 0.6, 1])

    with col2:
        if not st.session_state.correct_answer:
            with st.form(key="guess_form"):
                user_guess = st.text_input(
                    "Enter your guess (press Enter to submit):", key="user_guess_input"
                )
                submit_button = st.form_submit_button(
                    "Submit", use_container_width=True
                )

                if (
                    submit_button or user_guess
                ):  # This will trigger on Enter key or button click
                    try:
                        user_guess = int(user_guess)
                        if user_guess == vars[st.session_state.hold_out].evaluate(
                            st.session_state.problem
                        ):
                            st.session_state.correct_answer = True
                            st.success(
                                "Well done! Press Enter to move to the next question."
                            )
                            st.rerun()
                        else:
                            st.error("Try again!")
                    except ValueError:
                        st.error("Invalid input. Try again!")
        else:
            with st.form(key="next_question_form"):
                c1, c2 = st.columns([1, 10])
                with c1:
                    user_guess = st.text_input("", key="user_guess_input")
                with c2:
                    if st.form_submit_button(
                        "Next Question (Press Enter)", use_container_width=True
                    ):
                        st.session_state.problem = get_next_problem()
                        st.session_state.hold_out = random.choice([x, y, z]).name
                        st.session_state.correct_answer = False
                        st.rerun()

    js_code = """
    <script>
    function focusInput() {
        const inputs = window.parent.document.querySelectorAll('input[type="text"]');
        console.log(inputs)
        if (inputs.length > 0) {
            inputs[0].focus();
        }
    }
    
    // Initial focus
    focusInput();
    
    // Focus after each Streamlit rerun
    window.parent.addEventListener('DOMContentLoaded', focusInput);
    </script>
    """
    html(js_code)


if __name__ == "__main__":
    main()
