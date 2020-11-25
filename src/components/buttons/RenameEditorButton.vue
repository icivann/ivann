<template>
  <span @click="this.renameEditor">
    <i class="rename-button fa fa-pen"/>
  </span>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import { Getter, Mutation } from 'vuex-class';
import { uniqueTextInput } from '@/inputs/prompt';
import { EditorSave, saveEditor, SaveWithNames } from '@/file/EditorAsJson';
import EditorType from '@/EditorType';
import { EditorModel } from '@/store/editors/types';

@Component
export default class RenameEditorButton extends Vue {
  @Prop({ required: true }) readonly editorType!: EditorType;
  @Prop({ required: true }) readonly index!: number;
  @Prop({ required: true }) readonly oldName!: string;
  @Getter('editorNames') editorNames!: Set<string>;
  @Getter('saveWithNames') saveWithNames!: SaveWithNames;
  @Getter('editor') getEditor!: (editorType: EditorType, index: number) => EditorModel;
  @Getter('overviewEditor') overviewEditor!: EditorModel;
  @Mutation('renameEditor') rename!:
    (arg0: { editorType: EditorType; index: number; name: string }) => void;

  private renameEditor() {
    const name: string | null = uniqueTextInput(this.editorNames,
      'Please enter a unique name for the editor');
    if (name !== null) {
      this.rename({ editorType: this.editorType, index: this.index, name });

      // Re-save overview to get new node name
      const overviewEditorSave = saveEditor(this.overviewEditor);
      localStorage.setItem('unsaved-editor-Overview', JSON.stringify(overviewEditorSave));

      // Replace Local Storage with updated name
      const saved: EditorSave = saveEditor(this.getEditor(this.editorType, this.index));
      localStorage.setItem(`unsaved-editor-${name}`, JSON.stringify(saved));
      localStorage.setItem('unsaved-project', JSON.stringify(this.saveWithNames));
      localStorage.removeItem(`unsaved-editor-${this.oldName}`);
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
