import { Button as BaseButton } from "https://esm.sh/gh/jeff-hykin/good-component@0.3.0/elements.js"
import { Elemental, passAlongProps } from "https://esm.sh/gh/jeff-hykin/elemental@0.6.5/main/deno.js"
import { css } from "https://esm.sh/gh/jeff-hykin/good-component@0.3.0/elements.js"
import { addDynamicStyleFlags, createCssClass, setupClassStyles, hoverStyleHelper, combineClasses, mergeStyles, AfterSilent, removeAllChildElements } from "https://esm.sh/gh/jeff-hykin/good-component@0.3.0/main/helpers.js"
import { colors } from "./theme.js"

// 
// Button
// 
const buttonClass = css`
    background-color: ${colors.red};
    color: white; 
    padding: 10px 20px; 
    border: none; 
    border-radius: 4px; 
    font-size: 16px; 
    cursor: pointer; 
    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); 
    transition: all 0.3s ease;
    height: max-content;
    width: max-content;
    padding: 0.5rem 1rem;
    /* clean this up to be more pretty later */
    &:hover {
        opacity: 0.8;
        box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.3); /* Increased box-shadow on hover */
    }
    /* Focus effects for accessibility */
    &:focus {
        outline: none;
        box-shadow: 0 0 0 3px ${colors.blue}; /* Focus ring */
    }
`
export function Button({ ...props }) {
    const element = BaseButton({})
    globalThis.element = element
    props       = setupClassStyles(props)
    props.class = combineClasses(buttonClass, props.class)

    element.style.height = "max-content"
    element.style.width = "max-content"
    element.style.padding = "0.5rem 1rem"

    return passAlongProps(element,props)
}