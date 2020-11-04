<template>
  <span @click="this.renameEditor">
    <i class="rename-button fa fa-pen"/>
  </span>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import { Getter, Mutation } from 'vuex-class';
import { uniqueTextInput } from '@/inputs/prompt';
import EditorType from '../../EditorType';

@Component
export default class RenameEditorButton extends Vue {
  @Prop({ required: true }) readonly editorType!: EditorType;
  @Prop({ required: true }) readonly index!: number;
  @Getter('editorNames') editorNames!: Set<string>;
  @Mutation('renameEditor') rename!:
    (arg0: { editorType: EditorType; index: number; name: string}) => void;

  private renameEditor() {
    const name: string | null = uniqueTextInput(this.editorNames,
      'Please enter a unique name for the editor');
    if (name !== null) {
      this.rename({ editorType: this.editorType, index: this.index, name });
    }
  }
}
</script>

<style lang="scss" scoped>
  .rename-button {
    color: var(--foreground);
    background: none;
    padding: 0 4px 2px;
    border-radius: 2px;
    &:hover {
      background: var(--blue);
    }
  }
</style>
