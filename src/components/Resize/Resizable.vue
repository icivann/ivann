<template>
  <div class="resizable" @mousedown="resize">
    <slot/>
  </div>
</template>

<script lang="ts">
import { Component, Vue } from 'vue-property-decorator';

interface ResizableEvent extends MouseEvent {
  target: Element;
}

@Component({})
export default class Resizable extends Vue {
  private resize({ target, pageX: beforeX }: ResizableEvent) {
    if (target.className === 'resizer') {
      const first = target.previousElementSibling as HTMLStyleElement;
      const { offsetWidth: initialPaneWidth } = first;

      const resize = (before: number, offset = 0) => {
        const firstWidth = before + offset;
        this.$emit('width-change', firstWidth * 100 / this.$el.clientWidth);
      };

      const onMouseMove = (event: MouseEvent) => {
        resize(initialPaneWidth, event.pageX - beforeX);
      };

      const onMouseUp = () => {
        resize(first.clientWidth);

        window.removeEventListener('mousemove', onMouseMove);
        window.removeEventListener('mouseup', onMouseUp);
      };

      window.addEventListener('mousemove', onMouseMove);
      window.addEventListener('mouseup', onMouseUp);
    }
  }
}
</script>

<style scoped>
  .resizable {
    display: flex;
    flex-direction: row;
    user-select: none;
  }
</style>
