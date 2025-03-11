import { passAlongProps } from "https://esm.sh/gh/jeff-hykin/elemental@0.6.5/main/deno.js"
import * as fabric from "https://esm.sh/fabric@6.6.1/dist/fabric.min.js"
export { fabric }

// generic canvas component
export function FabricCanvas({
    width,
    height,
    backgroundColor,
    selectionColor,
    selectionLineWidth,
    onceFabricLoads,
    onMouseDown,
    onObjectAdded,
    onAfterRender,
    onMouseMove,
    onMouseUp,
    onBeforeSelection,
    onSelectionCreated,
    onSelectionCleared,
    onObjectModified,
    onObjectSelected,
    onObjectMoving,
    onObjectScaling,
    onObjectRotating,
    onObjectRemoved,
    jsonObjects,
    ...props
}={}) {
    const element = document.createElement("canvas")
    element.setAttribute("width", width||window.innerWidth)
    element.setAttribute("height", height||window.innerHeight)
    element.id = `${Math.random()}`
    element.style.position = "fixed"
    element.style.top = "0"
    element.style.left = "0"
    let intervalId = setInterval(async () => {
        // "when mounted to the dom" callback
        // NOTE: this is not performant. It would be REAL NICE if there was a callback for mounting to dom, but there isnt one
        //       so we have to poll for it
        if (element.parentElement) {
            clearInterval(intervalId)
            element.fabric = new fabric.Canvas(element.id)
            if (jsonObjects) {
                await element.fabric.loadFromJSON(jsonObjects)
            }
            Object.assign(element.fabric, {
                backgroundColor,
                selectionColor,
                selectionLineWidth,
            })
            console.debug(`element.fabric.backgroundColor is:`,element.fabric.backgroundColor)
            for (const [key, value] of Object.entries({
                "mouse:down": onMouseDown,
                "object:added": onObjectAdded,
                "after:render": onAfterRender,
                "mouse:move": onMouseMove,
                "mouse:up": onMouseUp,
                "before:selection": onBeforeSelection,
                "selection:created": onSelectionCreated,
                "selection:cleared": onSelectionCleared,
                "object:modified": onObjectModified,
                "object:selected": onObjectSelected,
                "object:moving": onObjectMoving,
                "object:scaling": onObjectScaling,
                "object:rotating": onObjectRotating,
                "object:removed": onObjectRemoved,
            })) {
                if (value) {
                    element.fabric.on(key, value)
                }
            }
            await (onceFabricLoads&&onceFabricLoads(element.fabric))
            element.fabric.renderAll()
        }
    }, 100)
    return passAlongProps(element, props)
}